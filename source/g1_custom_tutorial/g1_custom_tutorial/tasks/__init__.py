# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""この拡張に含まれるタスク実装群をまとめたパッケージ。"""

##
# Gym 環境を登録
##

from isaaclab_tasks.utils import import_packages

# ブラックリストはサブパッケージの設定をインポートしないために使用
_BLACKLIST_PKGS = ["utils", ".mdp"]
# このパッケージ配下の設定を一括インポート
import_packages(__name__, _BLACKLIST_PKGS)
