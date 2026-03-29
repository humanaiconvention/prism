import pandas as pd
import numpy as np

def analyze_parity(csv_path="logs/phase0/corrected/corrected_er.csv"):
    df = pd.read_csv(csv_path)
    
    print("\n" + "="*60)
    print("PHASE 7A: TEST 1 - EVEN/ODD LAYER PARITY")
    print("="*60)
    
    even_df = df[df['layer'] % 2 == 0]
    odd_df = df[df['layer'] % 2 == 1]
    
    metrics = ['block_output_er', 'post_norm_er', 'post_mixer_er', 'post_ffn_er']
    
    print(f"{'Metric':<18} | {'Even Mean':>10} | {'Odd Mean':>10} | {'Delta (Odd-Even)':>15}")
    print("-" * 62)
    for m in metrics:
        even_m = even_df[m].mean()
        odd_m = odd_df[m].mean()
        delta = odd_m - even_m
        print(f"{m:<18} | {even_m:>10.2f} | {odd_m:>10.2f} | {delta:>15.2f}")
        
    print("\n" + "="*60)
    print("PHASE 7A: TEST 2 - 4-LAYER BLOCK STRUCTURE (GLA/GLA/GLA/FoX)")
    print("="*60)
    
    # Genesis repeats FoX every 4 layers: [0,1,2]=GLA, [3]=FoX, [4,5,6]=GLA, [7]=FoX
    
    # Calculate mixer compression ratio: post_norm_er / post_mixer_er
    df['mixer_compression'] = df['post_norm_er'] / df['post_mixer_er']
    
    # Group layers by position in the 4-layer block
    df['block_pos'] = df['layer'] % 4
    
    positions = {
        0: "GLA (FoX-following)",
        1: "GLA (Mid)",
        2: "GLA (FoX-preceding)",
        3: "FoX (Reset layer)"
    }
    
    print(f"{'Position in Block':<25} | {'Block Out ER':>12} | {'Mixer Component Ratio (Norm/Mix)':>35}")
    print("-" * 78)
    for pos in range(4):
        pos_df = df[df['block_pos'] == pos]
        mean_er = pos_df['block_output_er'].mean()
        mean_comp = pos_df['mixer_compression'].mean()
        print(f"{positions[pos]:<25} | {mean_er:>12.2f} | {mean_comp:>35.2f}x")

if __name__ == "__main__":
    analyze_parity()
