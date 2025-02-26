# InSTA: Towards Internet-Scale Training For Agents

![Pipeline Overview](https://data-for-agents.github.io/static/images/pipeline_overview.png)

**Brandon Trabucco (1) Gunnar Sigurdsson (2) Robinson Piramuthu (2) Ruslan Salakhutdinov (1)**

**(1) Carnegie Mellon University, Machine Learning Department (2) Amazon**

The predominant approach for training web navigation agents gathers human demonstrations for a set of popular websites and hand-written tasks, but it is becoming clear that human data are an inefficient resource. We develop a pipeline to facilitate Internet-scale training for agents without laborious human annotations. In the first stage, an LLM generates tasks for 150k diverse websites. In the next stage, LLM agents complete tasks and produce trajectories. In the final stage, an LLM reviews the trajectories and judges their success. Language models are competitive with human annotators, detecting and filtering out harmful content with an accuracy of 97%, generating feasible tasks with an 89% rate, and judging successful trajectories with an 82.6% accuracy. Scaling the pipeline, agents based on Llama 3.1 70B solve 16.7% of tasks for 150k sites. Training on the data generated by our pipeline is competitive with training on human demonstrations. In data-limited settings derived from Mind2Web and WebLINX, we improve Step Accuracy by up to +89.5% and +122.1% respectively for agents trained on mixtures of data from our pipeline, and human data. When training agents with all available human data from these benchmarks, agents fail to generalize to diverse real sites, and adding our data improves their generalization by +149.0% for WebLINX and +156.3% for Mind2Web. Code available at: [data-for-agents.github.io](https://data-for-agents.github.io).

[website](https://data-for-agents.github.io)    |    [paper](https://arxiv.org/abs/2502.06776)    |    [data](https://huggingface.co/datasets/data-for-agents/insta-150k)

## Quickstart Guide

The quickest way to start is to load the environment with docker:

```bash
docker pull brandontrabucco/insta-browser-environment
docker run -p 7860:7860 -p 3000-3007:3000-3007 -t brandontrabucco/insta-browser-environment &
```

Then download and install the code for InSTA:

```bash
git clone https://github.com/data-for-agents/insta
cd insta && pip install -e .
```

And start vLLM to serve Llama 3.3 70B:

```bash
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
bash start_vllm_server.sh
```

Then, run the following example:

```python
from insta import (
    InstaPipeline,
    create_demo_videos
)

pipeline = InstaPipeline()

pipeline.run_pipeline(dataset = [
    {"domain": "example.com", "task": "example task"},
])

create_demo_videos(
    task_is_feasible_threshold = 0.0,
    success_threshold = 0.0,
    on_right_track_threshold = 0.0,
)
```

This example will run the `InstaPipeline` to collect agent trajectories for `example task` on `example.com`, and will save observations, actions, and evaluations to `./data`. Running the `create_demo_videos` function after running the `InstaPipeline` will visualize agent behaviors in the data by rendering MP4 videos, and accepts parameters for filtering trajectories by the estimated probability of success (see `success_threshold` above), and other conditions.

Videos are saved to `./data/videos` by default.

## Gym Environment & Tools

We are excited to present the **Official Gym Environment and LLM Tools** for using [InSTA](https://arxiv.org/abs/2502.06776) with popular LLM inference frameworks, including `transformers`, and `langchain`.

### Loading The Gym Environment

After pulling our docker image, starting the environment docker container, and starting a local vLLM server, you can load the `InstaEnv` and generate actions with a `BrowserAgent`:

```python
from insta import (
    InstaEnv,
    BrowserAgent
)

agent = BrowserAgent()

env = InstaEnv()

obs, info = env.reset(
    url = "http://example.com"
)

done = False

while not done:

    action = agent(
        observation = obs.processed_text,
        instruction = "example task"
    )

    obs, reward, done, truncated, info = obs.step(
        action = action
    )
```

### Loading Tools

We are serving a gradio demo at `http://insta.btrabuc.co:7860` for the demo tool below that allows you to start using InSTA without running the docker environment yourself.

```python
from insta import InstaTransformersGradioTool

tool = InstaTransformersGradioTool()

outputs = tool(
    url = "http://google.com"
)
```

Running the above will produce the following observation:

```
Here is your assigned session ID: `awesome-avocado`

You are visiting the URL: `http://google.com`

Here is the current viewport rendered in markdown:

Google [id: 4] About link [id: 5] Store link [id: 11] Gmail link [id: 13] Search for Images link [id: 16] Google apps link [id: 21] Sign in link Google image 
## Search Form
[id: 77] """

""" (q textbox)
[id: 89] Search by voice button
[id: 96] Search by image button
[id: 238] "Google Search" (btnK submit input)
[id: 239] "I'm Feeling Lucky" (btnI submit input) [id: 285] Advertising link [id: 286] Business link [id: 287] How Search works link [id: 289] data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABUAAAAYCAMAAAAiV0... link [id: 293] Privacy link [id: 294] Terms link [id: 300] Settings button
```

InSTA produces a compact markdown representation of webpages. We can represent typical webpages in as few as ~200 tokens (the demo above requires just 240 tokens). The `InstaTransformersGradioTool` uses a JSON-based action format by default, and we can fill the textbox marked `[id: 77]` with the following code snippet. The action format can be overridden to a Javascript format, and extended to custom formats depending on your needs.

```python
import json

action = json.dumps({
    "action_key": "fill",
    "action_kwargs": {
        "value": "latest meta llama models"
    },
    "target_element_id": 77
})

outputs = tool(
    session_id = "awesome-avocado",
    action = action
)
```

For LLM tools, frameworks like Langchain and Transformers assume they are stateless, so the session ID assigned by the tool must be propagated to future calls (note the `session_id = "awesome-avocado"` above). Running this action produces the following observation:

```
Here is your assigned session ID: `awesome-avocado`

You are visiting the URL: `http://google.com`

Here is the current viewport rendered in markdown:

Google [id: 4] About link [id: 5] Store link [id: 11] Gmail link [id: 13] Search for Images link [id: 16] Google apps link [id: 21] Sign in link Google image 
## Search Form
[id: 77] """
latest meta llama models
""" (q textbox)
[id: 83] Clear button
[id: 89] Search by voice button
[id: 96] Search by image button

* latest meta llama **model**
* llama **model** meta
* llama meta **demo**

[id: 325] "Google Search" (btnK submit input)
[id: 326] "I'm Feeling Lucky" (btnI submit input)
[id: 329] Report inappropriate predictions button
[id: 333] "Google Search" (btnK submit input)
[id: 334] "I'm Feeling Lucky" (btnI submit input) [id: 380] Advertising link [id: 381] Business link [id: 382] How Search works link [id: 384] data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABUAAAAYCAMAAAAiV0... link [id: 388] Privacy link [id: 389] Terms link [id: 395] Settings button
```

InSTA captures the structure, flow, hierarchy, and style of the webpage in its markdown representation. Interactive elements, including forms, buttons, links, and other widgets are noted with an `[id: ##]` identifier that agents can refer to.

## Citing Us

Please cite our work using the following bibtex:

```
@misc{Trabucco2025InSTA,
  title={InSTA: Towards Internet-Scale Training For Agents},
  author={Brandon Trabucco and Gunnar Sigurdsson and Robinson Piramuthu and Ruslan Salakhutdinov},
  year={2025},
  eprint={2502.06776},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
}
```