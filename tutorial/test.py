import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.decomposition import PCA, FactorAnalysis
from scipy.stats import zscore
import seaborn as sns
import pandas as pd
from pathlib import Path

# Add the smc_rnns repository to path
repo_path = Path("./smc_rnns")
if not repo_path.exists():
    print("Cloning the smc_rnns repository...")
    !git clone https://github.com/mackelab/smc_rnns.git
    
sys.path.append(str(repo_path))

# Import necessary modules from the repository
try:
    from smc_rnns.rnn import RNN
    from smc_rnns.rnn_models import ModelRNN
    from smc_rnns.smc import SMCParameters, SMC
    print("Successfully imported smc_rnns modules")
except ImportError as e:
    print(f"Error importing smc_rnns modules: {e}")
    print("Make sure the repository is properly installed")
    
def prepare_spike_data(spike_data, condition_labels, bin_size_ms=10):
    """
    Prepare spike data for analysis
    
    Parameters:
    - spike_data: numpy array of shape (m_trials, n_neurons, t_milliseconds)
        Contains binary spike data (0s and 1s)
    - condition_labels: array of shape (m_trials,)
        Contains condition labels (1-28) for each trial
    - bin_size_ms: int, optional
        Size of time bins in milliseconds for binning spikes
    
    Returns:
    - binned_data: numpy array
        Binned and rate-converted spike data
    - trial_info: DataFrame
        Information about each trial including condition
    """
    m_trials, n_neurons, t_ms = spike_data.shape
    
    # Bin the spike data if needed
    if bin_size_ms > 1:
        n_bins = t_ms // bin_size_ms
        binned_data = np.zeros((m_trials, n_neurons, n_bins))
        
        for i in range(n_bins):
            start_idx = i * bin_size_ms
            end_idx = (i + 1) * bin_size_ms
            # Sum spikes in each bin and convert to firing rate (spikes/bin)
            binned_data[:, :, i] = np.sum(spike_data[:, :, start_idx:end_idx], axis=2)
    else:
        binned_data = spike_data.copy()
    
    # Create trial info dataframe
    trial_info = pd.DataFrame({
        'trial_id': np.arange(m_trials),
        'condition': condition_labels
    })
    
    return binned_data, trial_info



def extract_low_rank_dynamics(spike_data, condition_labels, n_components=10):
    """
    Extract low-rank dynamics from neural spike data using PCA
    
    Parameters:
    - spike_data: numpy array of shape (m_trials, n_neurons, t_milliseconds)
    - condition_labels: array of shape (m_trials,)
    - n_components: int, number of components to extract
    
    Returns:
    - projections: dict of numpy arrays
        Low-dimensional projections for each condition
    - explained_variance: numpy array
        Explained variance ratios for the components
    - pca: PCA object
        Fitted PCA model
    """
    m_trials, n_neurons, t_ms = spike_data.shape
    
    # Prepare data
    binned_data, trial_info = prepare_spike_data(spike_data, condition_labels)
    
    # Reshape data for PCA: (trials*time, neurons)
    reshaped_data = binned_data.reshape(-1, n_neurons)
    
    # Z-score the data for better PCA results
    reshaped_data_z = zscore(reshaped_data, axis=0)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    projections_flat = pca.fit_transform(reshaped_data_z)
    
    # Reshape back to (trials, time, components)
    projections_all = projections_flat.reshape(m_trials, -1, n_components)
    
    # Organize projections by condition
    projections = {}
    for condition in np.unique(condition_labels):
        condition_mask = condition_labels == condition
        projections[f"condition_{condition}"] = projections_all[condition_mask]
    
    return projections, pca.explained_variance_ratio_, pca





def fit_smc_rnn(spike_data, condition_labels, latent_dim=10, hidden_dim=100):
    """
    Fit an SMC-RNN model to the spike data and extract latent trajectories
    
    Parameters:
    - spike_data: numpy array of shape (m_trials, n_neurons, t_milliseconds)
    - condition_labels: array of shape (m_trials,)
    - latent_dim: int, dimension of the latent space
    - hidden_dim: int, dimension of the RNN hidden state
    
    Returns:
    - model: fitted SMC-RNN model
    - latent_trajectories: dict of numpy arrays
        Inferred latent trajectories for each condition
    """
    try:
        m_trials, n_neurons, t_ms = spike_data.shape
        
        # Prepare data
        binned_data, trial_info = prepare_spike_data(spike_data, condition_labels)
        
        print("Setting up SMC-RNN model with parameters:")
        print(f"- Latent dimension: {latent_dim}")
        print(f"- Hidden dimension: {hidden_dim}")
        print(f"- Number of neurons: {n_neurons}")
        
        # Create initial RNN model
        rnn = RNN(
            input_size=n_neurons,  
            hidden_size=hidden_dim,
            output_size=n_neurons,
            latent_size=latent_dim
        )
        
        # Define SMC parameters - increase particles for better inference
        smc_params = SMCParameters(
            n_particles=200,  # More particles for better inference
            resample_threshold=0.5,
            noise_std=0.1
        )
        
        print("Creating ModelRNN and SMC objects...")
        # Create and fit the model - using the repository's models
        model = ModelRNN(rnn)
        smc = SMC(model, smc_params)
        
        # Convert data to format expected by SMC-RNN
        # (trials, time, neurons) format
        observations = np.transpose(binned_data, (0, 2, 1))
        
        print(f"Running SMC inference on {m_trials} trials...")
        # Run SMC for each trial and extract latent trajectories
        latent_trajectories = {}
        
        for condition in np.unique(condition_labels):
            print(f"Processing condition {condition}...")
            condition_mask = condition_labels == condition
            condition_obs = observations[condition_mask]
            
            # Combine trials for this condition
            condition_trajectories = []
            for trial_idx, trial_obs in enumerate(condition_obs):
                print(f"  Trial {trial_idx+1}/{len(condition_obs)}", end="\r")
                # Run SMC inference
                smc_results = smc.forward_filter(trial_obs)
                # Extract mean trajectory of latent variables
                mean_trajectory = smc_results.get_mean_trajectory()
                condition_trajectories.append(mean_trajectory)
            
            print(f"  Completed {len(condition_obs)} trials for condition {condition}")
            latent_trajectories[f"condition_{condition}"] = np.array(condition_trajectories)
        
        print("SMC-RNN inference completed successfully.")
        return model, latent_trajectories
    
    except Exception as e:
        print(f"Error in SMC-RNN fitting: {e}")
        print("Falling back to PCA-based analysis")
        return None, None

def analyze_smc_rnn_dynamics(spike_data, condition_labels, latent_dim=5):
    """
    Analyze neural dynamics using SMC-RNN instead of PCA
    
    Parameters:
    - spike_data: numpy array of shape (m_trials, n_neurons, t_milliseconds)
    - condition_labels: array of shape (m_trials,)
    - latent_dim: int, dimension of the latent space
    
    Returns:
    - model: fitted SMC-RNN model
    - latent_trajectories: dict of numpy arrays
    """
    print("=== SMC-RNN Analysis ===")
    print(f"Fitting SMC-RNN model with latent dimension {latent_dim}...")
    
    # Fit SMC-RNN model to full-rank spike data
    model, latent_trajectories = fit_smc_rnn(
        spike_data, 
        condition_labels, 
        latent_dim=latent_dim, 
        hidden_dim=max(100, latent_dim*10)  # Rule of thumb for hidden dimension
    )
    
    if model is None or latent_trajectories is None:
        print("SMC-RNN fitting failed. Falling back to PCA.")
        return None, None
    
    print("SMC-RNN model fitted successfully!")
    print(f"Extracted latent trajectories for {len(latent_trajectories)} conditions")
    
    # Visualize latent dynamics
    print("Visualizing SMC-RNN latent dynamics...")
    
    # Create placeholder explained variance (not applicable for SMC-RNN)
    placeholder_variance = np.ones(latent_dim) / latent_dim
    
    # Visualize the SMC-RNN latent dynamics
    visualize_dynamics(latent_trajectories, placeholder_variance, np.unique(condition_labels))
    
    # Visualize flow field from SMC-RNN latent dynamics
    print("Creating vector fields from SMC-RNN latent dynamics...")
    visualize_flow_field(latent_trajectories)
    
    # Perform fixed point analysis on the SMC-RNN latent space
    print("Analyzing fixed points in SMC-RNN latent dynamics...")
    analyze_fixed_points(latent_trajectories)
    
    return model, latent_trajectories

def analyze_fixed_points(projections):
    """
    Analyze fixed points in the neural dynamics
    
    Parameters:
    - projections: dict of numpy arrays
        Low-dimensional projections for each condition
    """
    # Estimate flow field
    X, Y, U, V = estimate_flow_field(projections, grid_size=40)  # Higher resolution
    
    # Compute vector magnitudes
    magnitude = np.sqrt(U**2 + V**2)
    
    # Create plot
    plt.figure(figsize=(14, 12))
    
    # Plot vector field with quiver for clearer direction visualization
    plt.quiver(X, Y, U, V, magnitude, cmap='viridis', scale=30, width=0.002)
    
    # Identify and mark fixed points (where U and V are close to zero)
    threshold = 0.05 * np.max(magnitude)
    potential_fixed_points = np.where(magnitude < threshold)
    
    fixed_points = []
    fixed_point_types = []
    
    if len(potential_fixed_points[0]) > 0:
        # Get coordinates of potential fixed points
        fixed_x = X[potential_fixed_points]
        fixed_y = Y[potential_fixed_points]
        
        # Cluster close points to find distinct fixed points
        from sklearn.cluster import DBSCAN
        if len(fixed_x) > 1:
            try:
                # Scale to ensure eps is appropriate
                scale = max(np.ptp(fixed_x), np.ptp(fixed_y))
                eps = 0.05 * scale if scale > 0 else 0.1
                
                clustering = DBSCAN(eps=eps, min_samples=1).fit(np.column_stack((fixed_x, fixed_y)))
                labels = clustering.labels_
                
                # Get cluster centers
                unique_labels = np.unique(labels)
                
                for label in unique_labels:
                    mask = labels == label
                    center_x = np.mean(fixed_x[mask])
                    center_y = np.mean(fixed_y[mask])
                    
                    # Analyze fixed point type by computing Jacobian
                    J = estimate_jacobian(X, Y, U, V, center_x, center_y)
                    fp_type = classify_fixed_point(J)
                    
                    fixed_points.append((center_x, center_y))
                    fixed_point_types.append(fp_type)
                    
                    # Plot fixed points with different markers based on type
                    if fp_type == "Stable Node":
                        plt.plot(center_x, center_y, 'go', markersize=12, label='Stable Node' if 'Stable Node' not in plt.gca().get_legend_handles_labels()[1] else "")
                    elif fp_type == "Unstable Node":
                        plt.plot(center_x, center_y, 'ro', markersize=12, label='Unstable Node' if 'Unstable Node' not in plt.gca().get_legend_handles_labels()[1] else "")
                    elif fp_type == "Saddle":
                        plt.plot(center_x, center_y, 'yo', markersize=12, label='Saddle Point' if 'Saddle Point' not in plt.gca().get_legend_handles_labels()[1] else "")
                    elif fp_type == "Center":
                        plt.plot(center_x, center_y, 'bo', markersize=12, label='Center' if 'Center' not in plt.gca().get_legend_handles_labels()[1] else "")
                    elif fp_type == "Spiral":
                        plt.plot(center_x, center_y, 'mo', markersize=12, label='Spiral' if 'Spiral' not in plt.gca().get_legend_handles_labels()[1] else "")
                    else:
                        plt.plot(center_x, center_y, 'ko', markersize=12, label='Fixed Point' if 'Fixed Point' not in plt.gca().get_legend_handles_labels()[1] else "")
            
            except Exception as e:
                print(f"Error in fixed point clustering: {e}")
                # Fallback if clustering fails
                plt.plot(fixed_x, fixed_y, 'ro', markersize=8, label='Potential Fixed Points')
        else:
            center_x, center_y = fixed_x[0], fixed_y[0]
            J = estimate_jacobian(X, Y, U, V, center_x, center_y)
            fp_type = classify_fixed_point(J)
            fixed_points.append((center_x, center_y))
            fixed_point_types.append(fp_type)
            plt.plot(center_x, center_y, 'ro', markersize=12, label=f'{fp_type}')
    
    # Plot trajectories
    condition_colors = plt.cm.tab20(np.linspace(0, 1, len(projections)))
    for i, (condition, proj) in enumerate(projections.items()):
        mean_trajectory = np.mean(proj, axis=0)
        plt.plot(mean_trajectory[:, 0], mean_trajectory[:, 1], 
                 color=condition_colors[i], linewidth=2, label=condition)
        plt.scatter(mean_trajectory[0, 0], mean_trajectory[0, 1], 
                   color=condition_colors[i], marker='o', s=100)
        plt.scatter(mean_trajectory[-1, 0], mean_trajectory[-1, 1], 
                   color=condition_colors[i], marker='x', s=100)
    
    plt.colorbar(label='Vector Magnitude')
    plt.title('Neural Dynamics Phase Portrait with Fixed Points')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.grid(True, alpha=0.3)
    
    # Add a legend only if fixed points were found
    if len(potential_fixed_points[0]) > 0:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
    plt.tight_layout()
    plt.show()
    
    # If fixed points were found, print their details
    if fixed_points:
        print("\nFixed Point Analysis:")
        print(f"Found {len(fixed_points)} fixed points in the dynamics\n")
        
        for i, ((fp_x, fp_y), fp_type) in enumerate(zip(fixed_points, fixed_point_types)):
            print(f"Fixed Point {i+1}:")
            print(f"  Location: ({fp_x:.3f}, {fp_y:.3f})")
            print(f"  Type: {fp_type}")
            print()

def estimate_jacobian(X, Y, U, V, x0, y0):
    """
    Estimate the Jacobian matrix at a fixed point
    
    Parameters:
    - X, Y: meshgrid coordinates
    - U, V: vector field components
    - x0, y0: coordinates of the fixed point
    
    Returns:
    - J: 2x2 Jacobian matrix
    """
    # Find closest grid point to the fixed point
    distances = (X - x0)**2 + (Y - y0)**2
    idx = np.unravel_index(np.argmin(distances), X.shape)
    
    # Get grid indices
    i, j = idx
    
    # Compute partial derivatives
    h = X[0, 1] - X[0, 0]  # Grid spacing
    
    # Handle boundary cases
    if i == 0:
        dUdy = (U[i+1, j] - U[i, j]) / h
        dVdy = (V[i+1, j] - V[i, j]) / h
    elif i == X.shape[0] - 1:
        dUdy = (U[i, j] - U[i-1, j]) / h
        dVdy = (V[i, j] - V[i-1, j]) / h
    else:
        dUdy = (U[i+1, j] - U[i-1, j]) / (2*h)
        dVdy = (V[i+1, j] - V[i-1, j]) / (2*h)
    
    if j == 0:
        dUdx = (U[i, j+1] - U[i, j]) / h
        dVdx = (V[i, j+1] - V[i, j]) / h
    elif j == X.shape[1] - 1:
        dUdx = (U[i, j] - U[i, j-1]) / h
        dVdx = (V[i, j] - V[i, j-1]) / h
    else:
        dUdx = (U[i, j+1] - U[i, j-1]) / (2*h)
        dVdx = (V[i, j+1] - V[i, j-1]) / (2*h)
    
    # Construct Jacobian
    J = np.array([[dUdx, dUdy], [dVdx, dVdy]])
    
    return J

def classify_fixed_point(J):
    """
    Classify fixed point based on its Jacobian
    
    Parameters:
    - J: 2x2 Jacobian matrix
    
    Returns:
    - classification: string describing the fixed point type
    """
    try:
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvals(J)
        real_parts = eigenvalues.real
        imag_parts = eigenvalues.imag
        
        # Check if all eigenvalues have zero real part (within numerical precision)
        if np.allclose(real_parts, 0, atol=1e-5):
            return "Center"
        
        # Check if any eigenvalues have imaginary parts
        if not np.allclose(imag_parts, 0, atol=1e-5):
            if np.all(real_parts < 0):
                return "Stable Spiral"
            elif np.all(real_parts > 0):
                return "Unstable Spiral"
            else:
                return "Spiral"
        
        # All eigenvalues are real
        if np.all(real_parts < 0):
            return "Stable Node"
        elif np.all(real_parts > 0):
            return "Unstable Node"
        else:
            return "Saddle"
    except:
        return "Unknown"

def main(spike_data, condition_labels, n_components=10, use_smc_rnn=True, latent_dim=5):
    """
    Main function to extract and visualize low-rank dynamics
    
    Parameters:
    - spike_data: numpy array of shape (m_trials, n_neurons, t_milliseconds)
    - condition_labels: array of shape (m_trials,)
    - n_components: int, number of components to extract for PCA
    - use_smc_rnn: bool, whether to use SMC-RNN for dynamics extraction
    - latent_dim: int, dimensionality of SMC-RNN latent space
    
    Returns:
    - either PCA or SMC-RNN results depending on use_smc_rnn parameter
    """
    print(f"Data shape: {spike_data.shape}")
    print(f"Number of conditions: {len(np.unique(condition_labels))}")
    
    if use_smc_rnn:
        print("\n=== Using SMC-RNN for low-rank dynamics extraction ===\n")
        model, latent_trajectories = analyze_smc_rnn_dynamics(
            spike_data, condition_labels, latent_dim=latent_dim
        )
        
        if model is not None and latent_trajectories is not None:
            return model, latent_trajectories
        else:
            print("SMC-RNN failed, falling back to PCA...")
    
    # If SMC-RNN is not used or failed, use PCA
    print("\n=== Using PCA for low-rank dynamics extraction ===\n")
    
    # Extract low-rank dynamics using PCA
    print("Extracting low-rank dynamics using PCA...")
    projections, explained_variance, pca = extract_low_rank_dynamics(
        spike_data, condition_labels, n_components
    )
    
    # Visualize results
    print("Visualizing dynamics and phase planes...")
    visualize_dynamics(projections, explained_variance, np.unique(condition_labels))
    
    # Visualize flow field
    print("Creating vector fields and flow visualizations...")
    visualize_flow_field(projections)
    
    # Analyze fixed points
    print("Analyzing fixed points in neural dynamics...")
    analyze_fixed_points(projections)
    
    return projections, explained_variance, pca

def visualize_dynamics(projections, explained_variance, condition_labels):
    """
    Visualize the extracted low-rank dynamics
    
    Parameters:
    - projections: dict of numpy arrays
        Low-dimensional projections for each condition
    - explained_variance: numpy array
        Explained variance ratios for the components
    - condition_labels: array of unique condition labels
    """
    # Plot explained variance
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Principal Components')
    plt.tight_layout()
    plt.show()
    
    # Plot first 3 PCs trajectories for each condition
    condition_colors = plt.cm.tab20(np.linspace(0, 1, len(projections)))
    
    # 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for i, (condition, proj) in enumerate(projections.items()):
        # Take mean across trials for cleaner visualization
        mean_trajectory = np.mean(proj, axis=0)
        ax.plot(
            mean_trajectory[:, 0], 
            mean_trajectory[:, 1], 
            mean_trajectory[:, 2],
            label=condition,
            color=condition_colors[i],
            linewidth=2
        )
        # Mark start point
        ax.scatter(
            mean_trajectory[0, 0],
            mean_trajectory[0, 1],
            mean_trajectory[0, 2],
            color=condition_colors[i],
            marker='o',
            s=100
        )
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('Neural Population Dynamics (First 3 PCs)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    # Plot first 2 PCs across time for different conditions
    plt.figure(figsize=(15, 10))
    
    for i, (condition, proj) in enumerate(projections.items()):
        mean_trajectory = np.mean(proj, axis=0)
        plt.subplot(1, 2, 1)
        plt.plot(mean_trajectory[:, 0], label=condition, color=condition_colors[i])
        plt.subplot(1, 2, 2)
        plt.plot(mean_trajectory[:, 1], label=condition, color=condition_colors[i])
    
    plt.subplot(1, 2, 1)
    plt.title('PC1 Activity over Time')
    plt.xlabel('Time Bin')
    plt.ylabel('PC1 Activity')
    
    plt.subplot(1, 2, 2)
    plt.title('PC2 Activity over Time')
    plt.xlabel('Time Bin')
    plt.ylabel('PC2 Activity')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    # Phase plane plots for selected condition pairs
    create_phase_plane_plots(projections, condition_colors)
    
    # Vector field visualization
    create_vector_field_plots(projections)

def create_phase_plane_plots(projections, condition_colors):
    """
    Create phase plane plots for the first two principal components
    
    Parameters:
    - projections: dict of numpy arrays
        Low-dimensional projections for each condition
    - condition_colors: array of colors for each condition
    """
    plt.figure(figsize=(15, 12))
    
    # Phase plane plot (PC1 vs PC2)
    plt.subplot(2, 2, 1)
    for i, (condition, proj) in enumerate(projections.items()):
        mean_trajectory = np.mean(proj, axis=0)
        plt.plot(
            mean_trajectory[:, 0], 
            mean_trajectory[:, 1],
            label=condition,
            color=condition_colors[i],
            linewidth=2
        )
        # Mark start point with circle and end point with X
        plt.scatter(
            mean_trajectory[0, 0], 
            mean_trajectory[0, 1],
            color=condition_colors[i],
            marker='o',
            s=100
        )
        plt.scatter(
            mean_trajectory[-1, 0], 
            mean_trajectory[-1, 1],
            color=condition_colors[i],
            marker='x',
            s=100
        )
        
        # Add arrows to show direction
        for t in range(0, len(mean_trajectory)-1, len(mean_trajectory)//10):  # Add arrows at intervals
            plt.arrow(
                mean_trajectory[t, 0], 
                mean_trajectory[t, 1],
                mean_trajectory[t+1, 0] - mean_trajectory[t, 0],
                mean_trajectory[t+1, 1] - mean_trajectory[t, 1],
                head_width=0.1,
                head_length=0.1,
                fc=condition_colors[i],
                ec=condition_colors[i]
            )
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Phase Plane: PC1 vs PC2')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Phase plane plot (PC2 vs PC3)
    plt.subplot(2, 2, 2)
    for i, (condition, proj) in enumerate(projections.items()):
        mean_trajectory = np.mean(proj, axis=0)
        plt.plot(
            mean_trajectory[:, 1], 
            mean_trajectory[:, 2],
            label=condition,
            color=condition_colors[i]
        )
        # Mark start and end points
        plt.scatter(
            mean_trajectory[0, 1], 
            mean_trajectory[0, 2],
            color=condition_colors[i],
            marker='o',
            s=100
        )
        plt.scatter(
            mean_trajectory[-1, 1], 
            mean_trajectory[-1, 2],
            color=condition_colors[i],
            marker='x',
            s=100
        )
        
    plt.xlabel('PC2')
    plt.ylabel('PC3')
    plt.title('Phase Plane: PC2 vs PC3')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Phase plane plot (PC1 vs PC3)
    plt.subplot(2, 2, 3)
    for i, (condition, proj) in enumerate(projections.items()):
        mean_trajectory = np.mean(proj, axis=0)
        plt.plot(
            mean_trajectory[:, 0], 
            mean_trajectory[:, 2],
            label=condition,
            color=condition_colors[i]
        )
        # Mark start and end points
        plt.scatter(
            mean_trajectory[0, 0], 
            mean_trajectory[0, 2],
            color=condition_colors[i],
            marker='o',
            s=100
        )
        plt.scatter(
            mean_trajectory[-1, 0], 
            mean_trajectory[-1, 2],
            color=condition_colors[i],
            marker='x',
            s=100
        )
        
    plt.xlabel('PC1')
    plt.ylabel('PC3')
    plt.title('Phase Plane: PC1 vs PC3')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Show legend in the fourth subplot space
    plt.subplot(2, 2, 4)
    plt.axis('off')
    plt.legend(loc='center', fontsize=12)
    
    plt.tight_layout()
    plt.show()

def create_vector_field_plots(projections, n_conditions_to_plot=5):
    """
    Create vector field visualizations for the neural dynamics
    
    Parameters:
    - projections: dict of numpy arrays
        Low-dimensional projections for each condition
    - n_conditions_to_plot: int, number of conditions to include in the plot
    """
    # Select a subset of conditions if there are too many
    if len(projections) > n_conditions_to_plot:
        conditions_to_plot = list(projections.keys())[:n_conditions_to_plot]
    else:
        conditions_to_plot = list(projections.keys())
    
    # Create a figure with subplots for each condition
    n_rows = int(np.ceil(len(conditions_to_plot) / 2))
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if len(conditions_to_plot) == 1 else axes
    
    for i, condition in enumerate(conditions_to_plot):
        ax = axes[i]
        proj = projections[condition]
        
        # Calculate mean trajectory
        mean_trajectory = np.mean(proj, axis=0)
        
        # Calculate velocity vectors
        velocity = np.diff(mean_trajectory, axis=0)
        points = mean_trajectory[:-1]  # Starting points for the vectors
        
        # Create a grid for the vector field
        x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
        margin = 0.1 * max(x_max - x_min, y_max - y_min)
        
        x = np.linspace(x_min - margin, x_max + margin, 20)
        y = np.linspace(y_min - margin, y_max + margin, 20)
        X, Y = np.meshgrid(x, y)
        
        # Interpolate vector field components
        from scipy.interpolate import griddata
        U = griddata((points[:, 0], points[:, 1]), velocity[:, 0], (X, Y), method='linear', fill_value=0)
        V = griddata((points[:, 0], points[:, 1]), velocity[:, 1], (X, Y), method='linear', fill_value=0)
        
        # Normalize vectors for better visualization
        magnitude = np.sqrt(U**2 + V**2)
        max_magnitude = np.max(magnitude)
        if max_magnitude > 0:
            U = U / max_magnitude
            V = V / max_magnitude
        
        # Plot trajectory
        ax.plot(mean_trajectory[:, 0], mean_trajectory[:, 1], 'k-', linewidth=2)
        ax.plot(mean_trajectory[0, 0], mean_trajectory[0, 1], 'go', markersize=8, label='Start')
        ax.plot(mean_trajectory[-1, 0], mean_trajectory[-1, 1], 'ro', markersize=8, label='End')
        
        # Plot vector field
        ax.quiver(X, Y, U, V, scale=25, width=0.002, color='blue', alpha=0.7)
        
        # Add streamplot for flow visualization
        try:
            ax.streamplot(X, Y, U, V, color='gray', linewidth=1, density=1, arrowstyle='->', arrowsize=1.5)
        except:
            pass  # Streamplot can fail with certain inputs
        
        ax.set_title(f'{condition} Vector Field')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
    
    # Hide any unused subplots
    for i in range(len(conditions_to_plot), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Create a combined vector field for multiple conditions
    plt.figure(figsize=(12, 10))
    plt.title('Combined Vector Fields')
    
    # Plot trajectories for each condition
    for i, condition in enumerate(conditions_to_plot):
        proj = projections[condition]
        mean_trajectory = np.mean(proj, axis=0)
        
        plt.plot(mean_trajectory[:, 0], mean_trajectory[:, 1], linewidth=2, label=condition)
        plt.scatter(mean_trajectory[0, 0], mean_trajectory[0, 1], marker='o', s=80)
        plt.scatter(mean_trajectory[-1, 0], mean_trajectory[-1, 1], marker='x', s=80)
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def estimate_flow_field(projections, grid_size=20):
    """
    Estimate a continuous flow field from discrete trajectories
    
    Parameters:
    - projections: dict of numpy arrays
        Low-dimensional projections for each condition
    - grid_size: int
        Size of the grid for vector field interpolation
    
    Returns:
    - X, Y: meshgrid coordinates
    - U, V: vector components at each grid point
    """
    # Combine all trajectories
    all_points = []
    all_velocities = []
    
    for condition, proj in projections.items():
        mean_trajectory = np.mean(proj, axis=0)
        velocity = np.diff(mean_trajectory, axis=0)
        points = mean_trajectory[:-1]  # Starting points for the vectors
        
        all_points.append(points)
        all_velocities.append(velocity)
    
    all_points = np.vstack(all_points)
    all_velocities = np.vstack(all_velocities)
    
    # Create a grid
    x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
    margin = 0.1 * max(x_max - x_min, y_max - y_min)
    
    x = np.linspace(x_min - margin, x_max + margin, grid_size)
    y = np.linspace(y_min - margin, y_max + margin, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Interpolate vector field
    from scipy.interpolate import griddata
    U = griddata((all_points[:, 0], all_points[:, 1]), all_velocities[:, 0], (X, Y), method='linear', fill_value=0)
    V = griddata((all_points[:, 0], all_points[:, 1]), all_velocities[:, 1], (X, Y), method='linear', fill_value=0)
    
    return X, Y, U, V

def visualize_flow_field(projections):
    """
    Visualize the estimated flow field of neural dynamics
    
    Parameters:
    - projections: dict of numpy arrays
        Low-dimensional projections for each condition
    """
    # Estimate flow field
    X, Y, U, V = estimate_flow_field(projections)
    
    # Compute vector magnitudes
    magnitude = np.sqrt(U**2 + V**2)
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Plot vector field with color indicating magnitude
    plt.streamplot(X, Y, U, V, density=1.5, color=magnitude, cmap='viridis', 
                  linewidth=1.5, arrowstyle='->', arrowsize=1.5)
    
    # Plot trajectories
    condition_colors = plt.cm.tab20(np.linspace(0, 1, len(projections)))
    for i, (condition, proj) in enumerate(projections.items()):
        mean_trajectory = np.mean(proj, axis=0)
        plt.plot(mean_trajectory[:, 0], mean_trajectory[:, 1], 
                 color=condition_colors[i], linewidth=2, label=condition)
        plt.scatter(mean_trajectory[0, 0], mean_trajectory[0, 1], 
                   color=condition_colors[i], marker='o', s=100)
        plt.scatter(mean_trajectory[-1, 0], mean_trajectory[-1, 1], 
                   color=condition_colors[i], marker='x', s=100)
    
    plt.colorbar(label='Vector Magnitude')
    plt.title('Neural Dynamics Flow Field (PC1 vs PC2)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()
    
    # Create a phase portrait
    plt.figure(figsize=(12, 10))
    
    # Plot vector field with quiver for clearer direction visualization
    plt.quiver(X, Y, U, V, magnitude, cmap='viridis', scale=30, width=0.002)
    
    # Identify and mark fixed points (where U and V are close to zero)
    threshold = 0.05 * np.max(magnitude)
    potential_fixed_points = np.where(magnitude < threshold)
    
    if len(potential_fixed_points[0]) > 0:
        # Get coordinates of potential fixed points
        fixed_x = X[potential_fixed_points]
        fixed_y = Y[potential_fixed_points]
        
        # Cluster close points (optional)
        from sklearn.cluster import DBSCAN
        if len(fixed_x) > 1:
            try:
                # Scale to ensure eps is appropriate
                scale = max(np.ptp(fixed_x), np.ptp(fixed_y))
                eps = 0.05 * scale if scale > 0 else 0.1
                
                clustering = DBSCAN(eps=eps, min_samples=1).fit(np.column_stack((fixed_x, fixed_y)))
                labels = clustering.labels_
                
                # Get cluster centers
                unique_labels = np.unique(labels)
                fixed_points = []
                
                for label in unique_labels:
                    mask = labels == label
                    center_x = np.mean(fixed_x[mask])
                    center_y = np.mean(fixed_y[mask])
                    fixed_points.append((center_x, center_y))
                
                # Plot fixed points
                for fp_x, fp_y in fixed_points:
                    plt.plot(fp_x, fp_y, 'ro', markersize=10, label='Fixed Point' if 'Fixed Point' not in plt.gca().get_legend_handles_labels()[1] else "")
            except:
                # Fallback if clustering fails
                plt.plot(fixed_x, fixed_y, 'ro', markersize=8, label='Potential Fixed Points')
        else:
            plt.plot(fixed_x, fixed_y, 'ro', markersize=10, label='Fixed Point')
    
    plt.colorbar(label='Vector Magnitude')
    plt.title('Neural Dynamics Phase Portrait with Fixed Points')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, alpha=0.3)
    
    # Add a legend only if fixed points were found
    if len(potential_fixed_points[0]) > 0:
        plt.legend()
        
    plt.tight_layout()
    plt.show()

def main(spike_data, condition_labels, n_components=10):
    """
    Main function to extract and visualize low-rank dynamics
    
    Parameters:
    - spike_data: numpy array of shape (m_trials, n_neurons, t_milliseconds)
    - condition_labels: array of shape (m_trials,)
    - n_components: int, number of components to extract
    """
    print(f"Data shape: {spike_data.shape}")
    print(f"Number of conditions: {len(np.unique(condition_labels))}")
    
    # Extract low-rank dynamics using PCA
    print("Extracting low-rank dynamics using PCA...")
    projections, explained_variance, pca = extract_low_rank_dynamics(
        spike_data, condition_labels, n_components
    )
    
    # Visualize results
    print("Visualizing dynamics and phase planes...")
    visualize_dynamics(projections, explained_variance, np.unique(condition_labels))
    
    # Visualize flow field
    print("Creating vector fields and flow visualizations...")
    visualize_flow_field(projections)
    
    # Try SMC-RNN if available
    print("Attempting to fit SMC-RNN model...")
    model, latent_trajectories = fit_smc_rnn(spike_data, condition_labels)
    
    if model is not None and latent_trajectories is not None:
        print("SMC-RNN model fitted successfully!")
        # Visualize SMC-RNN dynamics if available
        try:
            print("Visualizing SMC-RNN latent dynamics...")
            visualize_dynamics(latent_trajectories, np.ones(n_components)/n_components, np.unique(condition_labels))
            visualize_flow_field(latent_trajectories)
        except Exception as e:
            print(f"Could not visualize SMC-RNN latent dynamics: {e}")
    else:
        print("Using PCA results for analysis")
    
    return projections, explained_variance, pca

if __name__ == "__main__":
    # Generate sample data for demonstration
    # In practice, replace this with your actual data
    m_trials = 100
    n_neurons = 50
    t_ms = 1000
    n_conditions = 28
    
    # Generate random spike data (0s and 1s)
    spike_data = np.random.binomial(1, 0.05, size=(m_trials, n_neurons, t_ms))
    
    # Generate random condition labels (1 to 28)
    condition_labels = np.random.randint(1, n_conditions + 1, size=m_trials)
    
    # Run analysis with SMC-RNN instead of PCA
    model, latent_trajectories = main(
        spike_data, 
        condition_labels, 
        use_smc_rnn=True,  # Use SMC-RNN for latent dynamics
        latent_dim=5       # Lower dimensional latent space
    )