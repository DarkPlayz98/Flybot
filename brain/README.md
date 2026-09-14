# FlyWire brain bridge

FlyBot now has a real connectome adapter at `brain/flywire_bridge.py`.

## Architecture

```text
FlyWire FAFB connectome
        ↓
139k-neuron sparse LIF simulation
        ↓
sugar GRNs → proboscis motor neurons
        ↓
brain/flywire_bridge.py
        ↓ HTTPS + shared secret
FlyBot /brain/input
        ↓
Discord
```

The bridge uses the public `Tharusha101/flybrain` implementation as the neural
engine. That project loads the FlyWire adult Drosophila connectome and supports
single-timestep simulation through `Simulator.step()`. Its data files are not
bundled in GitHub and must be obtained according to that project's instructions.

## Install the neural engine

On the machine that runs the bridge:

```bash
git clone https://github.com/Tharusha101/flybrain vendor/flybrain
cd vendor/flybrain
pip install -r requirements.txt
```

Then obtain the required FlyWire data and build the sparse connectome cache as
described by the upstream project. The bridge expects the resulting cache at:

```text
vendor/flybrain/outputs/connectome
```

Set `FLYBRAIN_PATH` and `FLYBRAIN_CACHE` when using different locations.

## Bridge environment

```text
FLYBOT_URL=https://YOUR-FLYBOT-HOST
BRAIN_WEBHOOK_SECRET=the-same-secret-used-by-flybot
FLYBRAIN_PATH=./vendor/flybrain
FLYBRAIN_CACHE=./vendor/flybrain/outputs/connectome
FLYBRAIN_DEVICE=cpu
WEIGHT_SCALE=0.2
STIM_AMP=2.0
STEP_MS=40
PULSE_MS=250
ON_MS=25
```

Run:

```bash
python3 brain/flywire_bridge.py
```

## Important

This is a genuine connectome readout, but it is not a biological claim that the
fly is independently “thinking” in Discord. The bridge translates activity in a
specific motor-neuron pool into a Discord-friendly message. The current `feed`
output represents proboscis motor activity; it does not fabricate walking,
turning, or language outputs that are not provided by this brain model.
