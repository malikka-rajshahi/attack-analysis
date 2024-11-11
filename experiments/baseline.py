import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import json
from watermarking.generation import generate
from watermarking.detection import phi, fast_permutation_test
from watermarking.transform.key import transform_key_func
from watermarking.transform.sampler import transform_sampling

# Set up arguments
class Args:
    method = "transform"
    model = "../../attack-analysis/llama-3-8B-instruct"
    save = "1000_prompts_watermarked_text_results.json"
    seed = 0
    batch_size = 25  # Number of prompts processed per batch
    m = 10
    n = 256
    T = 1000  # Total number of prompts to process
    buffer_tokens = 20
    max_seed = 100000
    truncate_vocab = 8

args = Args()

# Seed for reproducibility
torch.manual_seed(args.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model)
#print("EOS token id:", tokenizer.eos_token_id)
#print("EOS token:", tokenizer.decode([tokenizer.eos_token_id]))
model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
vocab_size = model.get_output_embeddings().weight.shape[0]
eff_vocab_size = vocab_size - args.truncate_vocab
print(f'Loaded the model.')
torch.cuda.empty_cache()

# Dataset
dataset = load_dataset("c4", "default", split="train", streaming=True)

# Define the watermarking function to accept multiple prompts and seeds
def generate_watermark(prompts, seeds):
    return generate(
        model=model,
        prompts=prompts,
        vocab_size=vocab_size,
        n=args.n,
        m=args.m + args.buffer_tokens,
        seeds=seeds,
        key_func=transform_key_func,
        sampler=transform_sampling
    )

# Prepare null distribution for p-value calculation
def prepare_null_results(n, k, vocab_size, eff_vocab_size, n_runs=100):
    null_results = []
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    
    for _ in range(n_runs):
        torch.cuda.empty_cache()
        pi = torch.randperm(vocab_size)
        tokens = torch.argsort(pi)[:n]  # Random tokens for null distribution
        null_result = phi(
            tokens=tokens,
            n=n,
            k=10,
            generator=generator,
            key_func=transform_key_func,
            vocab_size=vocab_size,
            dist=lambda x, y: torch.norm(x - y),
            null=True,
            normalize=True
        )
        null_results.append(null_result)
    null_results = torch.sort(torch.tensor(null_results)).values
    return null_results

null_results = prepare_null_results(args.n, args.m, vocab_size, eff_vocab_size)

# Function for p-value calculation
def calculate_p_value(tokens, seed, n, k, vocab_size, null_results):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    test_stat = lambda tokens, n, k, generator, vocab_size, null: phi(
        tokens=tokens,
        n=n,
        k=10,
        generator=generator,
        key_func=transform_key_func,
        vocab_size=vocab_size,
        dist=lambda x, y: torch.norm(x - y),
        null=null,
        normalize=True
    )
    p_val = fast_permutation_test(
        tokens=tokens,
        vocab_size=vocab_size,
        n=n,
        k=10,
        seed=seed,
        test_stat=test_stat,
        null_results=null_results
    )
    return p_val

# Initialize data storage
results = []
# Prepare to iterate over the dataset
ds_iterator = iter(dataset)
batch_prompts = []
batch_size = args.batch_size

# Process each prompt and generate watermarked text in batches
for _ in range(args.T):
    torch.cuda.empty_cache()
    example = next(ds_iterator)
    text = example['text']
    prompt_text = text[:250]

    # Tokenize and adjust prompt length dynamically
    tokens = tokenizer.encode(prompt_text, return_tensors='pt', truncation=True, max_length=2048 - args.buffer_tokens)[0]

    # Set prompt and add to batch
    prompt = tokens
    batch_prompts.append(prompt)
    torch.cuda.empty_cache()

    # Once batch is filled, generate watermarked texts
    if len(batch_prompts) == batch_size:
        # Pad batch prompts and move to device
        batch_prompts_padded = pad_sequence(batch_prompts, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
        seeds = [torch.randint(high=args.max_seed, size=(1,)).item() for _ in range(batch_size)]
        
        # Generate watermarked samples
        watermarked_samples = generate_watermark(batch_prompts_padded, seeds)
        
        # Decode and process results for each prompt in the batch
        for i, watermarked_sample in enumerate(watermarked_samples):
            torch.cuda.empty_cache()
            prompt_len = batch_prompts[i].shape[0]
            generated_text_tokens = watermarked_sample[prompt_len:]  # Only the generated part
            watermarked_text = tokenizer.decode(generated_text_tokens, skip_special_tokens=True)
            
            # Calculate p-value using only the generated tokens
            p_value = calculate_p_value(
                generated_text_tokens,
                seeds[i],
                args.n,
                args.m,
                vocab_size,
                null_results
            )
            
            # Store result
            results.append({
                "prompt": tokenizer.decode(batch_prompts[i], skip_special_tokens=True),
                "watermarked_text": watermarked_text,
                "p_value": p_value.item()
            })

        # Reset batch
        batch_prompts = []

# Save results to JSON
with open(args.save, "w") as f:
    json.dump(results, f, indent=4)

print(f'Saved {len(results)} entries to {args.save}')

