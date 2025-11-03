"""
Base module for downloading and processing W&B run data.
"""

import wandb
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Optional, Any


class WandbDataLoader:
    """Handles downloading data from W&B API."""

    def __init__(self, entity: Optional[str] = None):
        """
        Initialize the data loader.

        Args:
            entity: W&B entity (username or team). If None, uses default.
        """
        self.api = wandb.Api()
        self.entity = entity

    def download_group_runs(
        self, project: str, group_name: str, metric_names: Optional[List[str]] = None, name_filter: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Download all runs from a specific group.

        Args:
            project: Name of the W&B project
            group_name: Name of the group to filter runs
            metric_names: List of metric names to download. If None, downloads all.
            name_filter: Filter runs by name. If None, downloads all. Can be a string to filter by a single name.
        Returns:
            DataFrame with run data
        """
        # Construct project path
        if self.entity:
            project_path = f"{self.entity}/{project}"
        else:
            project_path = project

        # Get runs filtered by group
        runs = self.api.runs(project_path, filters={"group": group_name, "name": {"$regex": name_filter}})

        print(f"Found {len(runs)} runs in group '{group_name}'")

        # Collect run data
        data = []
        for run in runs:
            run_data = {
                "run_id": run.id,
                "run_name": run.name,
                "group": run.group,
                "state": run.state,
                "created_at": run.created_at,
            }

            # Add config
            for key, value in run.config.items():
                run_data[f"config_{key}"] = value

            # Add summary metrics
            if metric_names:
                for metric in metric_names:
                    run_data[metric] = run.summary.get(metric, None)
            else:
                # Add all summary metrics
                for key, value in run.summary.items():
                    if not key.startswith("_"):
                        run_data[key] = value

            data.append(run_data)

        df = pd.DataFrame(data)
        print(f"Downloaded data shape: {df.shape}")
        return df

    def download_run_history(
        self, project: str, run_id: str, metric_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Download full history for a specific run.

        Args:
            project: Name of the W&B project
            run_id: ID of the specific run
            metric_names: List of metric names to download. If None, downloads all.

        Returns:
            DataFrame with full run history
        """
        if self.entity:
            project_path = f"{self.entity}/{project}"
        else:
            project_path = project

        run = self.api.run(f"{project_path}/{run_id}")

        # Get history as pandas DataFrame
        if metric_names:
            history = run.history(keys=metric_names)
        else:
            history = run.history()

        return history


class PostProcessor(ABC):
    """Abstract base class for post-processing W&B data."""

    @abstractmethod
    def process(self, df: pd.DataFrame) -> Any:
        """
        Process the downloaded data.

        Args:
            df: DataFrame containing run data

        Returns:
            Processed data (format depends on implementation)
        """
        pass

    @abstractmethod
    def save_results(self, results: Any, output_path: str) -> None:
        """
        Save processed results to disk.

        Args:
            results: Processed data
            output_path: Path to save results
        """
        pass
