import logging
from ast import literal_eval
from datetime import datetime
from os.path import abspath, exists, isabs, join
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from hydromt import hydromt_step
from hydromt.model.components import ModelComponent

from .config_variables import SfincsConfigVariables

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

logger = logging.getLogger(f"hydromt.{__name__}")


class SfincsConfig(ModelComponent):
    """ " Class to read and write SFINCS configuration files (sfincs.inp).

    This class provides methods for reading and writing SFINCS configuration files,
    updating configuration variables, and managing file paths for model input files.
    It uses a Pydantic model ([`SfincsConfigVariables`](hydromt_sfincs.components.config.config.SfincsConfigVariables))
    for validation and serialization of configuration variables.

    See Also
    --------
    :py:class:`~hydromt_sfincs.components.config.SfincsConfigVariables`
        Pydantic model class for SFINCS configuration variables.
    """

    def __init__(self, model: "SfincsModel"):
        self._filename = "sfincs.inp"
        self._data: SfincsConfigVariables = None
        super().__init__(model=model)

    @property
    def data(self):
        """Return the Pydantic SfincsConfigVariables object."""
        if self._data is None:
            self._data = SfincsConfigVariables()
            if self.root.is_reading_mode():
                self.read()
        return self._data

    @property
    def filename(self) -> str:
        """Return the filename of the SFINCS input file."""
        if not Path(self._filename).is_absolute():
            # If not absolute, join with the model root path
            root_path = self.model.root.path.resolve()
            self._filename = root_path / "sfincs.inp"
        return self._filename

    def read(self) -> None:
        """Read a text file with the sfincs configuration from the root folder and populate
        the SfincsConfigVariables. This function also determines the grid type and updates
        the grid properties of the SfincsModel (e.g. crs and extent)."""

        self.root._assert_read_mode

        if not exists(self.filename):
            raise FileNotFoundError(
                f"SFINCS input file '{self.filename}' does not exist."
            )

        # Read the file line by line
        with open(self.filename, "r") as fid:
            lines = fid.readlines()

        inp_dict = {}
        for line in lines:
            # Check if first character is #
            if line.strip().startswith("#"):
                # Full line comment
                continue
            # Find last character before #
            comment_idx = line.find("#")
            if comment_idx >= 0:
                line = line[:comment_idx]
            line = [x.strip() for x in line.split("=")]
            if len(line) != 2:
                continue
            name, val = line
            if name in ["tref", "tstart", "tstop"]:
                try:
                    val = datetime.strptime(val, "%Y%m%d %H%M%S")
                except ValueError:
                    ValueError(f'"{name} = {val}" not understood.')
            elif name in ["cdwnd", "cdval"]:
                val = [float(x) for x in val.split()]
            elif name == "utmzone":
                val = str(val)
            else:
                try:
                    val = literal_eval(val)
                except Exception:
                    pass

            if name == "crs":
                name = "epsg"

            inp_dict[name] = val

        # FIXME: when reading an existing config, you don't want to start with all possible variables?
        # Convert dictionary to SfincsConfig instance
        self._data = self.data.model_copy(update=inp_dict)

        # Update the grid properties from the configuration
        # This will either drop the quadtree component or the regular component?
        self.update_grid_from_config()

    def write(
        self, filename: str = "sfincs.inp", write_description: bool = False
    ) -> None:
        """Write the SfincsConfigVariables to a text file in the root folder of the model.

        Parameters:
        -----------
        filename (str):
            The name of the file to write the configuration to. Default is "sfincs.inp".
        write_description (bool):
            If True, include variable descriptions in the output file.  Default is False.
        """

        self.root._assert_write_mode

        if not isabs(filename) and self.root.path:
            self._filename = self.root.path / filename

        # Create parent directories if they do not exist
        self.filename.parent.mkdir(parents=True, exist_ok=True)

        # exclude_unset: Whether to exclude fields that have not been explicitly set.
        # exclude_defaults: Whether to exclude fields that are set to their default value.
        # exclude_none: Whether to exclude fields that have a value of `None`.
        # include: A set of fields to include in the output.
        # exclude: A set of fields to exclude from the output.

        with open(self.filename, "w") as fid:
            for key, value in self.data.model_dump(
                exclude_unset=False, exclude_defaults=False, exclude_none=True
            ).items():
                # Convert a value to a number if possible, otherwise return the original value
                value = convert_to_number(value)

                if isinstance(value, float):  # remove insignificant traling zeros
                    string = f"{key.ljust(20)} = {value}"
                elif isinstance(value, int):
                    string = f"{key.ljust(20)} = {value}"
                elif isinstance(value, list):
                    valstr = " ".join([str(v) for v in value])
                    string = f"{key.ljust(20)} = {valstr}"
                elif hasattr(value, "strftime"):
                    dstr = value.strftime("%Y%m%d %H%M%S")
                    string = f"{key.ljust(20)} = {dstr}"
                else:
                    string = f"{key.ljust(20)} = {value}"

                if key in self.data.__class__.model_fields:
                    description = self.data.__class__.model_fields[key].description
                    if description and write_description:
                        # Add description to string
                        string = string.ljust(50) + f" # {description}"

                fid.write(string + "\n")

    def get(self, key: str, fallback: Any = None, abs_path: bool = False) -> Any:
        """Get the value for a specific key with validation check.

        Parameters:
        -----------
        key (str):
            The key to retrieve the value for.
        fallback (Any):
            The fallback value to return if the key is not found. Default is None.
        abs_path (bool):
            If True and the value is a string or Path, return the absolute path.
        """

        value = self.data.model_dump().get(key, fallback)

        if value is None and fallback is not None:
            value = fallback
        if abs_path and isinstance(value, (str, Path)):
            value = Path(value)
            if not isabs(value):
                value = Path(abspath(join(self.root.path, value)))

        return value

    def set(self, key: str, value: Any, skip_validation: bool = False) -> None:
        """Set a value for a specific key with validation using Pydantic's model_validate.

        Parameters:
        -----------
        key (str):
            The key to set the value for.
        value (Any):
            The value to set.
        skip_validation (bool):
            If True, skips validation of the new value. Default is False, meaning pydantic validation will be performed.
            This checks amongst others for correct data types and valid ranges.
        """

        if not hasattr(self.data, key):
            raise KeyError(f"'{key}' is not a valid attribute of SfincsConfig.")

        if not skip_validation:
            # Merge full data to run full validation, including mode="before" for datetimes
            new_data = self.data.model_dump()
            new_data[key] = value
            self._data = self.data.__class__.model_validate(new_data)
        else:
            setattr(self._data, key, value)

    @hydromt_step
    def update(
        self,
        dict: Optional[Dict[str, Any]] = None,
        *,
        skip_validation: bool = False,
        **kwargs,
    ) -> None:
        """
        Update attributes using a dictionary or keyword arguments.

        Parameters:
        -----------
        dict (Dict[str, Any], optional):
            A dictionary containing key-value pairs to update the attributes.
            Example: dict = {'mmax': 100, 'nmax': 50}.
        skip_validation (bool, optional):
            If True, skips validation of the new values.
            Default is False, meaning pydantic validation will be performed.
            This checks amongst others for correct data types and valid ranges.
        kwargs:
            Key-value pairs passed as keyword arguments.
            Example: update(mmax=100, nmax=50)
        """
        updates = dict or {}
        updates.update(kwargs)

        if updates:
            logger.info(f"Updating {len(updates)} attributes in model config.")
            if skip_validation:
                # Bulk update without validation
                self._data = self._data.model_update(updates)
            else:
                new_data = self.data.model_dump()
                new_data.update(updates)
                self._data = self.data.__class__.model_validate(new_data)

    def update_grid_from_config(self) -> None:
        """Update the grid properties from the configuration. This method determines the grid type
        based on the presence of the 'qtrfile' variable in the configuration. If 'qtrfile' is set,
        the grid type is set to 'quadtree'; otherwise, it is set to 'regular'.

        Depending on the grid type, it updates the grid properties of the SfincsModel and removes
        the irrelevant grid component.
        """

        # Determine grid type based on configuration
        self.model.grid_type = "quadtree" if self.get("qtrfile") else "regular"

        if self.model.grid_type == "regular":
            # update the regular grid properties from the configuration
            self.model.grid.update_grid_from_config()
            # drop quadtree component
            for comp in self.model._QUADTREE_GRID_NAMES:
                self.model.components.pop(comp, None)
        elif self.model.grid_type == "quadtree":
            # drop regular component
            for comp in self.model._REGULAR_GRID_NAMES:
                self.model.components.pop(comp, None)

    def get_set_file_variable(
        self, key: str, value: str | Path = None, default: str = None
    ) -> Path:
        """
        Return the absolute file path for a given 'key'. If 'value' is provided,
        it is used and saved to config; otherwise, retrieves from config or uses default.

        Parameters:
        -----------
        key: str
            The config key, e.g., "obsfile"
        value: str | Path, optional
            Provided file name or path
        default: str, optional
            Default file name to use if no config value is found

        Recommended Usage:
        ------------------
        - For reading a file path from config:
            `get_set_file_variable("obsfile")`
        - For reading a custom file path and saving it to config:
            `get_set_file_variable("obsfile", value="sfincs_custom.obs")`
        - For setting a file path to config, always provide default value:
            `get_set_file_variable("obsfile", default="sfincs.obs")`

        Returns:
        --------
        Path: Absolute file path (not checked for existence)
        """

        root_path = self.model.root.path.resolve()

        # Convert to Path if needed
        if isinstance(value, str):
            value = Path(value)

        # If value is provided, use it and save to config
        if value is not None:
            # Input value is provided
            if not value.is_absolute():
                full_path = (root_path / value).resolve()
            else:
                full_path = value

            # Save to config (store relative name if under root)
            try:
                relative_path = full_path.relative_to(root_path)
                # NOTE In Python, if you want to convert a WindowsPath
                # object to a string without the double backslashes (\\),
                # you can use the as_posix() method instead of 'str'
                self.set(key, relative_path.as_posix())
            # If no relative path found, then use the full path:
            except ValueError:
                self.set(key, full_path.as_posix())

            return full_path

        # No value provided, try to get from config
        config_value = self.get(key)
        if config_value is not None:
            value_path = Path(config_value)
        # If config value is None, but default is provided:
        elif default is not None:
            value_path = Path(default)
            self.set(key, default)
        # If no value in config and no default provided:
        else:
            return None  # Nothing to return

        # Make sure the value is an absolute path
        if not value_path.is_absolute():
            return (root_path / value_path).resolve()
        else:
            return value_path


def convert_to_number(value):
    """Convert a value to a number if possible, otherwise return the original value."""
    try:
        if isinstance(value, str):
            value = value.strip()
        num = float(value)
        return int(num) if num.is_integer() else num
    except (ValueError, TypeError):
        return value
