import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dtaidistance import dtw
from joblib import Parallel, delayed
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sns.set_context("paper", font_scale=1.3)
RANDOM_SEED = 42


class TimeSeriesClusterer:
    def __init__(
        self,
        n_clusters: int = 15,  # Default to 15 clusters as in paper
        metric: str = "dtw",  # Dynamic Time Warping
        warping_window: float = 0.2,  # 20% warping window as mentioned in paper
        n_init: int = 5,
        max_iter: int = 75,
        tol: float = 1e-4,
        averaging_method: str = "dba",
        n_jobs: int = -1,
        random_state: int = RANDOM_SEED,
    ):
        """
        Initialize the Time Series Clusterer with enhanced parameters.

        Args:
            n_clusters (int): Number of clusters to create
            metric (str): Distance metric for clustering (default: DTW)
            warping_window (float): Size of warping window for DTW (proportion of series length)
            n_init (int): Number of times the algorithm runs with different centroid seeds
            max_iter (int): Maximum number of iterations for clustering
            tol (float): Tolerance to declare convergence
            averaging_method (str): Method for averaging time series
            n_jobs (int): Number of parallel jobs to run (-1 for all cores)
            random_state (int): Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.metric = metric
        self.warping_window = warping_window
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.averaging_method = averaging_method
        self.n_jobs = n_jobs
        self.random_state = random_state

        # Results storage
        self.labels_ = None
        self.cluster_centers_ = None
        self.inertia_ = None
        self.X = None
        self.series_ids = None
        self.id_to_index = {}

        # For optimization
        self.optimization_results: dict[str, list[int | float]] = {
            "n_clusters": [],
            "inertia": [],
            "silhouette_scores": [],
        }

        # Try importing tslearn for DBA if needed
        try:
            import tslearn.barycenters

            self._has_tslearn = True
        except ImportError:
            self._has_tslearn = False
            if averaging_method == "dba":
                print("Warning: tslearn not found. Using mean for averaging instead.")
                self.averaging_method = "mean"

    def _compute_dtw_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute DTW distance between two time series.

        Args:
            x (np.ndarray): First time series
            y (np.ndarray): Second time series

        Returns:
            float: DTW distance between x and y
        """
        # Convert warping_window from percentage to actual window size
        window = int(self.warping_window * len(x))

        # Use dtaidistance for faster computation
        return dtw.distance(x, y, window=window, use_pruning=True)

    def _compute_dba_center(self, series_cluster):
        """
        Compute DBA (DTW Barycenter Averaging) for a cluster.

        Args:
            series_cluster (np.ndarray): Time series in a cluster

        Returns:
            np.ndarray: Centroid series
        """
        if self._has_tslearn and self.averaging_method == "dba":
            from tslearn.barycenters import dtw_barycenter_averaging

            # Create metric_params dictionary with the constraint
            metric_params = {"sakoe_chiba_radius": int(self.warping_window * series_cluster.shape[1])}

            # Get the DBA center
            center = dtw_barycenter_averaging(
                series_cluster,
                max_iter=10,
                barycenter_size=series_cluster.shape[1],
                metric_params=metric_params,
            )

            # Ensure the center has the right shape by squeezing out any extra dimensions
            return center.squeeze()
        else:
            # Fallback to mean if tslearn is not available
            return np.mean(series_cluster, axis=0)

    def preprocess_data(self, X: np.ndarray) -> np.ndarray:
        """
        Preprocess time series data by standardizing (z-score normalization)

        Args:
            X (np.ndarray): Input time series data

        Returns:
            np.ndarray: Standardized time series data
        """
        return zscore(X, axis=1)

    def fit(self, X: np.ndarray, series_ids: list[str] | None = None) -> "TimeSeriesClusterer":
        """
        Fit the clusterer to the data

        Args:
            X (np.ndarray): Input time series data
            series_ids (List[str], optional): Unique identifiers for each time series

        Returns:
            TimeSeriesClusterer: Fitted clusterer
        """
        # Preprocess data
        self.X = self.preprocess_data(X)
        self.series_ids = series_ids if series_ids is not None else [f"series_{i}" for i in range(X.shape[0])]

        # Map IDs to indices
        self.id_to_index = {id_: idx for idx, id_ in enumerate(self.series_ids)}

        np.random.seed(self.random_state)

        # Step 1: Initialize cluster centers using K-means++ on flattened data
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=self.n_init,
            max_iter=1,  # Only for initialization
            random_state=self.random_state,
            init="k-means++",
        )
        flattened_X = self.X.reshape(self.X.shape[0], -1)
        initial_labels = kmeans.fit_predict(flattened_X)

        # Initialize cluster centers using DBA
        centers = np.zeros((self.n_clusters, self.X.shape[1]))
        for k in range(self.n_clusters):
            mask = initial_labels == k
            if mask.sum() > 0:
                centers[k] = self._compute_dba_center(self.X[mask])
            else:
                # Handle empty cluster
                centers[k] = self.X[np.random.choice(self.X.shape[0])]

        # Step 2: Iterative optimization
        labels = initial_labels
        old_inertia = float("inf")

        for iteration in range(self.max_iter):
            # Assign points to nearest center using DTW
            distances = np.zeros((self.X.shape[0], self.n_clusters))

            # Compute distances in parallel
            def compute_distances_to_centers(i):
                series_distances = np.zeros(self.n_clusters)
                for k in range(self.n_clusters):
                    series_distances[k] = self._compute_dtw_distance(self.X[i], centers[k])
                return series_distances

            all_distances = Parallel(n_jobs=self.n_jobs)(
                delayed(compute_distances_to_centers)(i) for i in range(self.X.shape[0])
            )
            distances = np.array(all_distances)

            # Assign to nearest center
            new_labels = np.argmin(distances, axis=1)

            # Update centers using DBA
            new_centers = np.zeros_like(centers)
            for k in range(self.n_clusters):
                mask = new_labels == k
                if mask.sum() > 0:
                    new_centers[k] = self._compute_dba_center(self.X[mask])
                else:
                    # Handle empty cluster
                    new_centers[k] = centers[k]

            # Calculate inertia (sum of squared DTW distances to closest center)
            inertia = 0
            for i, label in enumerate(new_labels):
                inertia += distances[i, label] ** 2

            # Check convergence
            if np.array_equal(labels, new_labels) or abs(old_inertia - inertia) < self.tol:
                break

            labels = new_labels
            centers = new_centers
            old_inertia = inertia

            if iteration % 10 == 0:
                print(f"Iteration {iteration}, inertia: {inertia:.4f}")

        # Store results
        self.labels_ = labels
        self.cluster_centers_ = centers
        self.inertia_ = inertia

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster membership for new data.

        Args:
            X (np.ndarray): Input time series data

        Returns:
            np.ndarray: Cluster labels
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted yet. Call fit() before predict().")

        X_processed = self.preprocess_data(X)

        def assign_to_cluster(series):
            distances = np.array([self._compute_dtw_distance(series, center) for center in self.cluster_centers_])
            return np.argmin(distances)

        labels = Parallel(n_jobs=self.n_jobs)(
            delayed(assign_to_cluster)(X_processed[i]) for i in range(X_processed.shape[0])
        )

        return np.array(labels)

    def get_label_from_id(self, series_id: str) -> int:
        """
        Get cluster label for a specific series ID

        Args:
            series_id (str): Unique identifier of the time series

        Returns:
            int: Cluster label
        """
        if series_id not in self.id_to_index:
            raise ValueError(f"Series ID {series_id} not found")
        idx = self.id_to_index[series_id]
        return self.labels_[idx]

    def plot_clusters(self, max_series_per_cluster: int = 10, save_path: str | None = None):
        """
        Plot clusters with centroids and sample series from each cluster in a grid layout.
        Only bottom-most plots in each column have x-axis labels, and only leftmost plots
        in each row have y-axis labels. Uses a single legend at the bottom.

        Args:
            max_series_per_cluster (int): Maximum number of series to plot per cluster
            save_path (str, optional): Path to save the plot
        """
        if self.labels_ is None or self.cluster_centers_ is None:
            raise ValueError("Model not fitted yet. Call fit() before plot_clusters().")

        # Define distinct colors for centroids
        centroid_colors = sns.color_palette("colorblind", self.n_clusters)

        # Calculate grid dimensions
        cols = min(3, self.n_clusters)  # Max 3 columns
        rows = (self.n_clusters + cols - 1) // cols  # Ceiling division to get rows

        # Create a single figure with subplots
        fig, axes = plt.subplots(rows, cols, figsize=(10, 12), squeeze=False)

        # Flatten the axes array for easier indexing
        axes_flat = axes.flatten()

        # Create empty lists to store handles and labels for the legend
        legend_handles = []
        legend_labels = []

        # Loop through each cluster
        for i in range(self.n_clusters):
            # Calculate grid position (row, col) from flattened index
            row, col = i // cols, i % cols
            ax = axes[row, col]

            # Plot series in this cluster
            cluster_series = self.X[self.labels_ == i]

            # Limit number of series plotted
            cluster_series = cluster_series[:max_series_per_cluster]

            # Plot individual series in gray
            # Create a line for sample members (only add to legend from the first subplot)
            if i == 0:
                # Plot one series and save its handle for the legend
                sample_line = ax.plot(
                    cluster_series[0] if len(cluster_series) > 0 else [],
                    color="gray",
                    alpha=0.4,
                )[0]
                legend_handles.append(sample_line)
                legend_labels.append("Cluster members")

                # Plot the rest without adding to legend
                for series in cluster_series[1:]:
                    ax.plot(series, color="gray", alpha=0.4)
            else:
                # Plot all series without adding to legend
                for series in cluster_series:
                    ax.plot(series, color="gray", alpha=0.4)

            # Plot centroid in color
            centroid_line = ax.plot(
                self.cluster_centers_[i],
                color=centroid_colors[i % len(centroid_colors)],
                linewidth=3,
            )[0]

            # Only add the first centroid to the legend
            if i == 0:
                legend_handles.append(centroid_line)
                legend_labels.append("Cluster centroid")

            # Set title for all plots
            ax.set_title(f"Cluster {i}")

            # Only show x-labels for bottom-most plots in each column or for the last plot
            is_bottom_plot = row == rows - 1 or (col == i % cols and (i + cols) >= self.n_clusters)
            if is_bottom_plot:
                ax.set_xlabel("Week")
            else:
                ax.set_xlabel("")

            # Only show y-labels for leftmost plots in each row
            if col == 0:
                ax.set_ylabel("Standardized Flow")
            else:
                ax.set_ylabel("")

            # Add horizontal dashed line at y=0
            ax.axhline(0, color="black", linestyle="--", linewidth=0.5)

            # Remove individual legends
            # ax.legend()
            ax.grid(True, alpha=0.3)
            sns.despine(ax=ax)

        # Hide unused subplots
        for j in range(self.n_clusters, len(axes_flat)):
            fig.delaxes(axes_flat[j])

        plt.subplots_adjust(bottom=0.1)

        # Add a single legend at the bottom of the figure
        fig.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc="lower center",
            ncol=2,
            bbox_to_anchor=(0.5, 0),
            frameon=True,
        )

        # Add padding at the bottom for the legend
        plt.subplots_adjust(bottom=0.1)
        plt.tight_layout()

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.tight_layout()
        # Adjust after tight_layout to make room for the legend
        plt.subplots_adjust(bottom=0.1)
        plt.show()

    def describe_clusters(self) -> list[dict]:
        """
        Provide descriptive statistics for each cluster

        Returns:
            List[dict]: Descriptive statistics for each cluster
        """
        if self.labels_ is None:
            raise ValueError("Model not fitted yet. Call fit() before describe_clusters().")

        cluster_descriptions = []
        for i in range(self.n_clusters):
            cluster_data = self.X[self.labels_ == i]
            cluster_desc = {
                "cluster_id": i,
                "num_series": len(cluster_data),
                "mean_series": np.mean(cluster_data, axis=0),
                "std_series": np.std(cluster_data, axis=0),
            }
            cluster_descriptions.append(cluster_desc)

        return cluster_descriptions

    def optimize_clusters(
        self, X: np.ndarray, min_clusters: int = 4, max_clusters: int = 20
    ) -> dict[str, list[int | float]]:
        """
        Find optimal number of clusters using inertia and silhouette score.

        Args:
            X (np.ndarray): Preprocessed time series data
            min_clusters (int): Minimum number of clusters to try
            max_clusters (int): Maximum number of clusters to try

        Returns:
            Dict with optimization results
        """
        X_processed = self.preprocess_data(X)

        # Reset optimization results
        self.optimization_results = {
            "n_clusters": list(range(min_clusters, max_clusters + 1)),
            "inertia": [],
            "silhouette_scores": [],
        }

        for n_clusters in self.optimization_results["n_clusters"]:
            print(f"Testing {n_clusters} clusters...")

            # Create and fit clusterer
            temp_clusterer = TimeSeriesClusterer(
                n_clusters=n_clusters,
                metric=self.metric,
                warping_window=self.warping_window,
                n_init=self.n_init,
                max_iter=self.max_iter,
                tol=self.tol,
                averaging_method=self.averaging_method,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )

            # Fit and get labels
            temp_clusterer.fit(X_processed)
            labels = temp_clusterer.labels_

            # Store inertia
            self.optimization_results["inertia"].append(temp_clusterer.inertia_)

            # Calculate silhouette score (only if more than one cluster)
            if n_clusters > 1:
                # Flatten the time series for silhouette score calculation
                X_flat = X_processed.reshape(X_processed.shape[0], -1)
                sil_score = silhouette_score(X_flat, labels)
                self.optimization_results["silhouette_scores"].append(sil_score)
            else:
                self.optimization_results["silhouette_scores"].append(0)

        return self.optimization_results

    def plot_cluster_optimization(self, save_path: str = None):
        """
        Plot elbow curve and silhouette scores for cluster optimization.
        """
        if not self.optimization_results["n_clusters"]:
            raise ValueError("Run optimize_clusters() first")

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Elbow curve (Inertia)
        ax1.plot(
            self.optimization_results["n_clusters"],
            self.optimization_results["inertia"],
            marker="o",
        )
        ax1.set_xlabel("Number of Clusters")
        ax1.set_ylabel("Inertia")
        ax1.set_title("Elbow Method")

        # Silhouette scores
        ax2.plot(
            self.optimization_results["n_clusters"],
            self.optimization_results["silhouette_scores"],
            marker="o",
            color="red",
        )
        ax2.set_xlabel("Number of Clusters")
        ax2.set_ylabel("Silhouette Score")
        ax2.set_title("Silhouette Analysis")

        sns.despine(ax=ax1)
        sns.despine(ax=ax2)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.tight_layout()
        plt.show()

    def recommend_clusters(self, method: str = "elbow") -> int:
        """
        Recommend optimal number of clusters.

        Args:
            method (str): Method to use for recommendation ('elbow' or 'silhouette')

        Returns:
            int: Recommended number of clusters
        """
        if not self.optimization_results["n_clusters"]:
            raise ValueError("Run optimize_clusters() first")

        if method == "elbow":
            # Find the "elbow" point where the rate of decrease in inertia slows down
            inertia = self.optimization_results["inertia"]
            # Calculate the rate of change
            inertia_diff = np.diff(inertia)
            elbow_index = np.argmax(np.abs(inertia_diff)) + 1
            return self.optimization_results["n_clusters"][elbow_index]

        elif method == "silhouette":
            # Find the number of clusters with the highest silhouette score
            sil_scores = self.optimization_results["silhouette_scores"]
            max_sil_index = np.argmax(sil_scores)
            return self.optimization_results["n_clusters"][max_sil_index]

        else:
            raise ValueError("Method must be 'elbow' or 'silhouette'")
