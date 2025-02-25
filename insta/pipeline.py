from typing import Callable, Tuple, List, Dict, Generator
from collections import namedtuple

from insta import (
    DEFAULT_AGENT_CONFIG,
    DEFAULT_JUDGE_CONFIG,
    DEFAULT_BROWSER_CONFIG,
    AgentConfig,
    JudgeConfig,
    BrowserConfig,
    InstaEnv,
    BrowserAgent,
    BrowserJudge,
    NULL_ACTION,
    NULL_JUDGMENT,
)

from datasets import load_dataset

import random
import tqdm

import json
import os

DEFAULT_DATASET = "data-for-agents/insta-150k"
DEFAULT_DATASET_SPLIT = "train"

DEFAULT_OBSERVATIONS_DIR = "data/observations"
DEFAULT_SCREENSHOT_DIR = "data/screenshots"
DEFAULT_ACTIONS_DIR = "data/actions"
DEFAULT_JUDGMENTS_DIR = "data/judgments"

DEFAULT_MAX_ACTIONS = 30
DEFAULT_SEED = 123
DEFAULT_SKIP_FINISHED = False

DEFAULT_RANK = 0
DEFAULT_WORLD_SIZE = 1


METADATA_KEYS = [
    'backend_node_id',
    'bounding_client_rect',
    'computed_style',
    'scroll_left',
    'scroll_top',
    'editable_value'
]


InstaPipelineOutput = namedtuple(
    "InstaPipelineOutput",
    ["observations", "actions", "judgment"]
)


class InstaPipeline(Callable):
    """Initialize the InSTA pipeline for internet-scale data collection,
    creates a browser, LLM agent, and LLM judge, then runs the agent
    to attempt web navigation tasks from the InSTA-150k dataset.

    """

    agent: BrowserAgent = None
    judge: BrowserJudge = None
    env: InstaEnv = None

    def __init__(self, agent_config: AgentConfig = DEFAULT_AGENT_CONFIG,
                 judge_config: JudgeConfig = DEFAULT_JUDGE_CONFIG,
                 browser_config: BrowserConfig = DEFAULT_BROWSER_CONFIG,
                 dataset: str = DEFAULT_DATASET,
                 dataset_split: str = DEFAULT_DATASET,
                 observations_dir: str = DEFAULT_OBSERVATIONS_DIR,
                 screenshot_dir: str = DEFAULT_SCREENSHOT_DIR,
                 actions_dir: str = DEFAULT_ACTIONS_DIR,
                 judgments_dir: str = DEFAULT_JUDGMENTS_DIR,
                 max_actions: int = DEFAULT_MAX_ACTIONS,
                 seed: int = DEFAULT_SEED,
                 skip_finished: bool = DEFAULT_SKIP_FINISHED,
                 rank: int = DEFAULT_RANK,
                 world_size: int = DEFAULT_WORLD_SIZE):
        """Initialize the InSTA pipeline for internet-scale data collection,
        creates a browser, LLM agent, and LLM judge, then runs the agent
        to attempt web navigation tasks from the InSTA-150k dataset.

        Arguments:

        agent_config: AgentConfig
            Configuration for the LLM agent.

        judge_config: JudgeConfig
            Configuration for the LLM judge.

        browser_config: BrowserConfig
            Configuration for the Playwright environment.

        dataset: str
            Dataset id to load from Huggingface.

        observations_dir: str
            Directory to save observations.

        screenshot_dir: str
            Directory to save screenshots.

        actions_dir: str
            Directory to save actions.

        judgments_dir: str
            Directory to save judgments.

        max_actions: int
            Maximum number of actions per task.

        seed: int
            Seed for the dataset.

        skip_finished: bool
            Whether to skip tasks that are already attempted.

        rank: int
            Rank of the process.

        world_size: int
            Number of data collection processes.
        
        """

        self.agent_config = agent_config
        self.judge_config = judge_config
        self.browser_config = browser_config

        self.observations_dir = observations_dir
        self.screenshot_dir = screenshot_dir
        self.actions_dir = actions_dir
        self.judgments_dir = judgments_dir

        self.max_actions = max_actions
        self.skip_finished = skip_finished
        self.seed = seed

        self.rank = rank
        self.world_size = world_size

        os.makedirs(
            self.observations_dir, 
            exist_ok = True
        )

        os.makedirs(
            self.screenshot_dir, 
            exist_ok = True
        )

        os.makedirs(
            self.actions_dir, 
            exist_ok = True
        )

        os.makedirs(
            self.judgments_dir, 
            exist_ok = True
        )

        self.dataset = load_dataset(
            dataset, split = dataset_split
        )

        self.agent = BrowserAgent(
            config = self.agent_config
        )

        self.judge = BrowserJudge(
            config = self.judge_config
        )

        self.env = InstaEnv(
            config = self.browser_config
        )

    def generate(self, url: str, instruction: str) -> Tuple[List[Dict], List[Dict]]:
        """Attempt a web navigation task using the LLM agent, and return the
        observations and actions along the trajectory for later processing.

        Arguments:

        url: str
            Starting URL for the agent.

        instruction: str
            Specific instruction for the agent.

        Returns:

        Tuple[List[Dict], List[Dict]]
            Tuple containing observations and actions from along the trajectory
            generated by running the agent with an instruction.
        
        """
    
        observations = []
        actions = []

        action = NULL_ACTION
    
        for action_id in range(self.max_actions):

            if action_id == 0:  # reset the envirnoment

                self.agent.reset_context()

                outputs = self.env.reset(
                    url = url
                )

            else:  # add action to the context

                if action is NULL_ACTION: break

                self.agent.push_action(
                    response = action.response
                )

                outputs = self.env.step(
                    action = action
                )

                if outputs.done: break

            obs = outputs.observation
            metadata = None

            if obs.metadata is not None:

                metadata = {
                    backend_node_id: {
                        key: node_metadata.get(key)
                        for key in METADATA_KEYS
                    } for backend_node_id, node_metadata in
                    obs.metadata.items()
                }

            observations.append({
                "current_url": obs.current_url,
                "processed_text": obs.processed_text,
                "raw_html": obs.raw_html,
                "screenshot": obs.screenshot,
                "metadata": metadata
            })
    
            self.agent.pop_observation()
            
            action = self.agent(
                observation = obs.processed_text,
                instruction = instruction
            )

            function_calls = [
                {"dotpath": x.dotpath, "args": x.args}
                for x in action.function_calls
            ]

            actions.append({
                "function_calls": function_calls,
                "response": action.response,
                "matched_response": action.matched_response
            })

        return observations, actions
    
    def __call__(self, url: str, instruction: str) -> Tuple[List[Dict], List[Dict]]:
        """Attempt a web navigation task using the LLM agent, and return the
        observations and actions along the trajectory for later processing.

        Arguments:

        url: str
            Starting URL for the agent.

        instruction: str
            Specific instruction for the agent.

        Returns:

        Tuple[List[Dict], List[Dict]]
            Tuple containing observations and actions from along the trajectory
            generated by running the agent with an instruction.
        
        """
        
        return self.generate(
            url = url,
            instruction = instruction
        )

    def iter_pipeline(self) -> Generator[InstaPipelineOutput, None, None]:
        """Run the InSTA pipeline for internet-scale data collection, and
        yield the observations, actions, and judgments for each task.

        Returns:

        Generator[InstaPipelineOutput, None, None]
            Generator for the observations, actions, and judgments for each task, 
            which are saved to disk for later processing.
        
        """

        dataset_ids = list(range(len(self.dataset)))

        random.seed(self.seed)
        random.shuffle(dataset_ids)

        dataset_ids = dataset_ids[
            self.rank::self.world_size
        ]

        progress_bar = tqdm.tqdm(
            dataset_ids, desc = "Processing",
            dynamic_ncols = True
        )

        for example_id in progress_bar:

            example_dict = self.dataset[example_id]
            domain = example_dict["domain"]
            task = example_dict["task"]

            progress_bar.set_description(
                "Processing: {}".format(
                    domain
                )
            )

            observations_path = os.path.join(
                self.observations_dir,
                "{}.json".format(domain)
            )

            actions_path = os.path.join(
                self.actions_dir,
                "{}.json".format(domain)
            )

            judgments_path = os.path.join(
                self.judgments_dir,
                "{}.json".format(domain)
            )

            screenshot_dir = os.path.join(
                self.screenshot_dir,
                "{}".format(domain)
            )

            os.makedirs(
                screenshot_dir,
                exist_ok = True
            )

            skip_this_task = (
                self.skip_finished and
                os.path.exists(judgments_path)
            )

            if skip_this_task:

                continue
            
            observations, actions = self.generate(
                url = "http://{}".format(domain),
                instruction = task
            )

            for observation_id, observation in enumerate(observations):

                screenshot = observation.pop("screenshot")

                if screenshot is not None:

                    screenshot_path = os.path.join(
                        screenshot_dir,
                        "screenshot_{:02d}.jpg"
                        .format(observation_id)
                    )

                    screenshot.convert("RGB").save(
                        screenshot_path
                    )

                    observation["screenshot_path"] = (
                        screenshot_path
                    )

            judgment = self.judge(
                observations = [
                    x["processed_text"]
                    for x in observations
                ],
                actions = [
                    x["response"]
                    for x in actions
                ],
                instruction = task
            )

            judgment = {
                "task_is_feasible": judgment.values.get(
                    "task_is_feasible"
                ),
                "success": judgment.values.get(
                    "success"
                ),
                "on_right_track": judgment.values.get(
                    "on_right_track"
                ),
                "response": judgment.response,
                "matched_response": judgment.matched_response,
            }
                    
            with open(observations_path, "w") as file:
                
                json.dump(
                    observations, 
                    file,
                    indent = 4
                )

            with open(actions_path, "w") as file:
                
                json.dump(
                    actions, 
                    file,
                    indent = 4
                )

            with open(judgments_path, "w") as file:
                
                json.dump(
                    judgment, 
                    file,
                    indent = 4
                )

            yield InstaPipelineOutput(
                observations = observations,
                actions = actions,
                judgment = judgment
            )

            
    def run_pipeline(self) -> None:
        """Run the InSTA pipeline for internet-scale data collection, and
        yield the observations, actions, and judgments for each task.

        """

        for x in self.iter_pipeline():
            
            pass
