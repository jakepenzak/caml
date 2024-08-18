import ray
from ray.util.spark import (
    MAX_NUM_WORKER_NODES,
    setup_ray_cluster,
    shutdown_ray_cluster,
)


def setup_ray_cluster_on_spark(**setup_ray_cluster_kwargs):
    """
    Setup a Ray cluster on Spark.
    """
    try:
        shutdown_ray_cluster()
    except RuntimeError:
        pass
    setup_ray_cluster(max_worker_nodes=MAX_NUM_WORKER_NODES, **setup_ray_cluster_kwargs)

    ray.init(address="auto", ignore_reinit_error=True)
