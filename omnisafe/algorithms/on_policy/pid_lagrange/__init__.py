# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""PID Lagrange algorithms."""

from omnisafe.algorithms.on_policy.pid_lagrange.cppo_pid import CPPOPID
from omnisafe.algorithms.on_policy.pid_lagrange.trpo_pid import TRPOPID
from omnisafe.algorithms.on_policy.pid_lagrange.ppo_pid import PPOPID
from omnisafe.algorithms.on_policy.pid_lagrange.rpo_pid import RPOPID


__all__ = [
    'CPPOPID',
    'TRPOPID',
    'PPOPID',
    'RPOPID',
]
