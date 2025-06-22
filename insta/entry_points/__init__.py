from insta.entry_points.insta_pipeline import (
    get_agent_config_from_cli,
    get_judge_config_from_cli,
    get_task_proposer_config_from_cli,
    get_dataset_from_cli,
    get_data_dirs_from_cli,
    get_pipeline_from_cli,
    launch_pipeline_from_cli,
    start_insta_pipeline
)

from insta.entry_points.annotate_task_proposer import (
    annotate_task_proposer_from_cli,
    start_annotate_task_proposer
)

from insta.entry_points.annotate_judge import (
    annotate_judge_from_cli,
    start_annotate_judge
)