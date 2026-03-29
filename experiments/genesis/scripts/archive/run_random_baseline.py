import numpy as np
import scipy.linalg
from tqdm import tqdm

def random_orthogonal_matrix(n):
    # Generating a random orthogonal matrix via QR decomposition
    H = np.random.randn(n, n)
    Q, R = np.linalg.qr(H)
    return Q

def get_random_subspace(d, k):
    Q = random_orthogonal_matrix(d)
    return Q[:, :k]

def main():
    d = 576  # Genesis-152M hidden dimension
    k = 185  # Effective Rank cut-off
    
    print(f"Running Random Subspace Baseline (d={d}, k={k})")
    
    num_trials = 100
    all_angles = []
    
    for _ in tqdm(range(num_trials), desc="Simulations"):
        subspace1 = get_random_subspace(d, k)
        subspace2 = get_random_subspace(d, k)
        
        # scipy.linalg.subspace_angles gives radians
        angles_rad = scipy.linalg.subspace_angles(subspace1, subspace2)
        angles_deg = np.degrees(angles_rad)
        all_angles.append(angles_deg)
        
    all_angles = np.array(all_angles)
    
    mean_angles = np.mean(all_angles, axis=0)
    
    print("\n--- Random Subspace Baseline Results (Averaged over 100 trials) ---")
    print(f"Mean Principal Angle: {np.mean(mean_angles):.2f} degrees")
    print(f"Median Principal Angle: {np.median(mean_angles):.2f} degrees")
    print(f"Min Angle across all trials: {np.min(all_angles):.2f} degrees")
    
    num_orthogonal = np.sum(mean_angles > 80.0)
    percent_orthogonal = (num_orthogonal / k) * 100
    print(f"Dimensions > 80° apart: {num_orthogonal}/{k} ({percent_orthogonal:.1f}%)")
    
    num_aligned = np.sum(mean_angles < 30.0)
    print(f"Dimensions < 30° apart (shared base): {num_aligned}/{k}")
    
    # Save the baseline comparison logic
    with open("measurements/random_baseline_principal_angles.txt", "w") as f:
        f.write(f"Random Baseline for Principal Angles (d={d}, k={k})\n")
        f.write("====================================================\n")
        f.write(f"Mean Angle: {np.mean(mean_angles):.2f} degrees\n")
        f.write(f"Median Angle: {np.median(mean_angles):.2f} degrees\n")
        f.write(f"Min Angle across all 100 trials: {np.min(all_angles):.2f} degrees\n")
        f.write(f"Average dimensions > 80 degrees apart: {num_orthogonal}/{k} ({percent_orthogonal:.1f}%)\n")
        f.write(f"Average dimensions < 30 degrees apart: {num_aligned}/{k}\n")
        
    print("\nSaved baseline data to measurements/random_baseline_principal_angles.txt")

if __name__ == "__main__":
    main()
