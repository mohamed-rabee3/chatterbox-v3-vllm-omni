# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chatterbox Multilingual V3 on vLLM-Omni.

Two native stages:

* ``chatterbox_mtl_v3_t3``    (``LLM_AR``)          text + conditioning -> speech codec ids
* ``chatterbox_mtl_v3_s3gen`` (``LLM_GENERATION``)  codec ids -> 24 kHz waveform

See ``constants.py`` for every pinned architectural value and ``vendor/`` for
the vendored official model modules.
"""
