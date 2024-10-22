
import einops as ei
import argparse
import datetime
import functools as ft
import os
import pathlib
import ipdb
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import yaml
import scipy

from gcbfplus.algo import GCBF, GCBFPlus, make_algo, CentralizedCBF, DecShareCBF
from gcbfplus.env import make_env
from gcbfplus.env.base import RolloutResult
from gcbfplus.trainer.utils import get_bb_cbf
from gcbfplus.utils.graph import GraphsTuple
from gcbfplus.utils.utils import jax_jit_np, tree_index, chunk_vmap, merge01, jax_vmap
from gcbfplus.env.obstacle import Obstacle, Rectangle

# ROS
import rospy
from gcbfplus_interface.msg import Barrier
from gcbfplus_interface.srv import Barriers, BarriersRequest

# Obstacle type
RECTANGLE = jnp.zeros(1)
CUBOID = jnp.ones(1)
SPHERE = jnp.ones(1) * 2


def generate_env_rviz(obstacle: Obstacle):
    rospy.init_node('obstacle_pub', anonymous=True)
    # 请求端
    obstacle_client = rospy.ServiceProxy('/Obstacle', Barriers)
    print("Waiting for obstacle service")
    rospy.wait_for_service('/Obstacle')

    barriers_msg: BarriersRequest = BarriersRequest()
    for i in range(obstacle.type.shape[0]):
        if obstacle.type[i][0] == RECTANGLE:
            barrier_msg: Barrier = Barrier()
            barrier_msg.barrier_type = int(obstacle.type[i][0])
            barrier_msg.center.x = obstacle.center[i][0]
            barrier_msg.center.y = obstacle.center[i][1]
            barrier_msg.width = obstacle.width[i]
            barrier_msg.height = obstacle.height[i]
            barrier_msg.theta = obstacle.theta[i]
            # barrier_msg.points = [item for item in obstacle.points[i].tolist()]
        elif obstacle.type[i][0] == CUBOID:
            barrier_msg: Barrier = Barrier()
            barrier_msg.barrier_type = int(obstacle.type[i][0])
            barrier_msg.center.x = obstacle.center[i][0]
            barrier_msg.center.y = obstacle.center[i][1]
            barrier_msg.center.z = obstacle.center[i][2]
            barrier_msg.length = obstacle.length[i]
            barrier_msg.width = obstacle.width[i]
            barrier_msg.height = obstacle.height[i]
            barrier_msg.rotation = obstacle.rotation[i]
            # barrier_msg.points = [item for item in obstacle.points[i].tolist()]
        elif obstacle.type[i][0] == SPHERE:
            barrier_msg: Barrier = Barrier()
            barrier_msg.barrier_type = int(obstacle.type[i][0])
            barrier_msg.center.x = obstacle.center[0][i]
            barrier_msg.center.y = obstacle.center[1][i]
            barrier_msg.center.z = obstacle.center[2][i]
            barrier_msg.radius = obstacle.radius[i]
        barriers_msg.barriers.append(barrier_msg)
        print(barriers_msg)
    resp = obstacle_client.call(barriers_msg)
    if resp.success:
        print("成功生成环境")
    else:
        print("生成环境失败")


params = {
    "car_radius": 0.05,
    "comm_radius": 0.5,
    "n_rays": 32,
    "obs_len_range": [0.1, 0.6],
    "n_obs": 8,
}

def test(args):
    # print(f"> Running test.py {args}")

    np.random.seed(args.seed)

    # load config
    if not args.u_ref and args.path is not None:
        with open(os.path.join(args.path, "config.yaml"), "r") as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)
    
     # create environments
    num_agents = config.num_agents if args.num_agents is None else args.num_agents
    env = make_env(
        env_id=config.env if args.env is None else args.env,
        num_agents=num_agents,
        num_obs=args.obs,
        area_size=args.area_size,
        max_step=args.max_step,
        max_travel=args.max_travel,
    )

    # TODO:测试生成环境

    test_key = jr.PRNGKey(args.seed)
    test_keys = jr.split(test_key, 1_000)[: args.epi]
    test_keys = test_keys[args.offset:]
    for i_epi in range(args.epi):
        key_x0, _ = jr.split(test_keys[i_epi], 2)
        n_rng_obs = args.obs
        assert n_rng_obs >= 0
        obstacle_key, key = jr.split(key_x0, 2)
        obs_pos = jr.uniform(obstacle_key, (n_rng_obs, 2), minval=0, maxval=args.area_size)
        length_key, key = jr.split(key, 2)
        obs_len = jr.uniform(
            length_key,
            (args.obs, 2),
            minval=params["obs_len_range"][0],
            maxval=params["obs_len_range"][1],
        )
        theta_key, key = jr.split(key, 2)
        obs_theta = jr.uniform(theta_key, (n_rng_obs,), minval=0, maxval=2 * np.pi)
        obstacles = env.create_obstacles(obs_pos, obs_len[:, 0], obs_len[:, 1], obs_theta)
    
        print(obstacles)
        generate_env_rviz(obstacles)
        print("Finished generate environment in rviz")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-agents", type=int, default=None)
    parser.add_argument("--obs", type=int, default=2)
    parser.add_argument("--area-size", type=float, default=4)
    parser.add_argument("--max-step", type=int, default=256)
    parser.add_argument("--path", type=str, default="pretrained/SingleIntegrator/gcbf+")
    parser.add_argument("--n-rays", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-travel", type=float, default=None)
    parser.add_argument("--cbf", type=int, default=None)

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--u-ref", action="store_true", default=False)
    parser.add_argument("--env", type=str, default="SingleIntegrator")
    parser.add_argument("--algo", type=str, default="gcbf+")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--epi", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--no-video", action="store_true", default=False)
    parser.add_argument("--nojit-rollout", action="store_true", default=False)
    parser.add_argument("--log", action="store_true", default=False)
    parser.add_argument("--dpi", type=int, default=100)

    args = parser.parse_args()
    test(args)


if __name__ == "__main__":
    main()