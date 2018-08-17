import random
import copy
import matplotlib.pyplot as plt

class ClusterGenerator:
    """Generates test clusters for the k-Means-Algorithm"""

    @staticmethod
    def generate_cluster(mean, sigma, num):
        """Generate a gaussian distributed cluster of points."""
        dims = len(mean)
        assert len(sigma) == dims

        cluster = []
        for _ in range(num):
            point = tuple(random.gauss(mean[i], sigma[i]) for i in range(dims))
            cluster.append(point)
        return cluster

    @staticmethod
    def generate_dataset(num_clusters, dims=2,
                         mean_range=(-10.0, 10.0), sigma_range=(0.5, 1.5), num_range=(25, 100)):
        """Generate a set of gaussian distributed clusters of points."""
        dataset = []
        for _ in range(num_clusters):
            mean = tuple(random.uniform(*mean_range) for _ in range(dims))
            sigma = tuple(random.uniform(*sigma_range) for _ in range(dims))
            num = random.randint(*num_range)
            dataset += ClusterGenerator.generate_cluster(mean, sigma, num)
        return dataset


class KMeans:
    """Simple implementation of the standard k-Means-Algorithm"""

    def __init__(self, dataset, k):
        self.dataset = dataset
        self.num_points = len(dataset)
        self.k = k
        self.dims = len(self.dataset[0])

        assert self.num_points >= self.k

        self.select_random_means()
        self.assign_points_to_clusters()

        former_means = []
        while former_means != self.cluster_means:
            former_means = copy.copy(self.cluster_means)
            self.calculate_new_means()
            self.assign_points_to_clusters()

    def select_random_means(self):
        """Select k random points as a starting mean value."""
        self.cluster_means = []

        mean_indices = []
        for i in range(self.k):
            index = random.randint(0, self.num_points - 1 - i)
            # make sure the indices are distinct
            for j in range(i):
                if index > mean_indices[j]:
                    index += 1

            mean_indices.append(index)
            self.cluster_means.append(self.dataset[index])

    def assign_points_to_clusters(self):
        """Calculate which cluster every point belongs to."""
        self.point_clusters = []
        for point in self.dataset:
            self.point_clusters.append(self.select_closest_cluster_index(point))

    def select_closest_cluster_index(self, point):
        """Select the closest cluster mean for a single point."""
        # calculate the squared distance to a certain mean -> our function to minimize
        sqr_distance = lambda index: sum(
            (point[i] - self.cluster_means[index][i])**2
            for i in range(self.dims)
        )
        return min(range(self.k), key=sqr_distance)

    def calculate_new_means(self):
        """Calculate new cluster means based on which points currently belongs to each cluster."""
        new_means = []
        for i in range(self.k):
            cluster_points = self.get_points_of_cluster(i)
            if cluster_points:
                new_means.append(self.calculate_mean(cluster_points))
            else: # handle the rare case that there are no points for a cluster
                new_means.append(self.cluster_means[i])
        self.cluster_means = new_means

    def get_points_of_cluster(self, cluster_index):
        """Get all points which currently belong to a cluster."""
        points = []
        for i in range(self.num_points):
            if self.point_clusters[i] == cluster_index:
                points.append(self.dataset[i])
        return points

    def calculate_mean(self, points):
        """Calculate a mean value of a set of points."""
        num_points = len(points)
        mean = [0.0] * self.dims
        for point in points:
            for i in range(self.dims):
                mean[i] += point[i]
        for i in range(self.dims):
            mean[i] /= num_points
        return mean

    def plot(self):
        """Show a (2D) plot of the dataset with colored clusters."""
        plt.figure(figsize=(7, 7))
        for i in range(self.k):
            points = self.get_points_of_cluster(i)

            x_vals = tuple(point[0] for point in points)
            y_vals = tuple(point[1] for point in points)

            plt.plot(x_vals, y_vals, '.C{}'.format(i))

        plt.tight_layout(pad=0.2)
        plt.show()


if __name__ == '__main__':
    def test():
        """Test the k-Means-Algorithm"""
        dims = 2

        num_clusters = 5
        mean_range = (-10.0, 10.0)
        sigma_range = (1.0, 2.0)
        num_range = (10, 150)

        dataset = ClusterGenerator.generate_dataset(num_clusters, dims,
                                                    mean_range, sigma_range, num_range)
        kmeans = KMeans(dataset, num_clusters)
        kmeans.plot() # note: we only plot 2 dimensions

    test()
