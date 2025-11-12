# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to print available environments in this extension.

Tip: pass --search to filter by ID substring (e.g., "Custom-", "Template-", "Isaac-").
By default it searches for "Custom-" since this repo registers Custom-* IDs.
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import argparse
import gymnasium as gym
from prettytable import PrettyTable

import g1_custom_tutorial.tasks  # noqa: F401


def main():
    """Print all environments registered in `g1_custom_tutorial` extension."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", type=str, default="Custom-", help="Substring to match in Gym IDs")
    args = parser.parse_args()
    # print all the available environments
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available Environments in Isaac Lab"
    # set alignment of table columns
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    # count of environments
    index = 0
    # acquire all Isaac environments names
    registry = getattr(gym, "registry", gym.envs.registry)
    for task_spec in registry.values():
        if args.search in task_spec.id:
            # add details to table
            cfg_key = "env_cfg_entry_point" if "env_cfg_entry_point" in task_spec.kwargs else task_spec.kwargs.get("env_cfg_entry_point", "-")
            table.add_row([index + 1, task_spec.id, task_spec.entry_point, cfg_key])
            # increment count
            index += 1

    print(table)


if __name__ == "__main__":
    try:
        # run the main function
        main()
    except Exception as e:
        raise e
    finally:
        # close the app
        simulation_app.close()
