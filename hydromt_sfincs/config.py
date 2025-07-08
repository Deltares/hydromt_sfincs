from datetime import datetime
import time
from typing import TYPE_CHECKING, List, Optional, Dict, Any
from ast import literal_eval
from os.path import abspath, isabs, join, split, exists
from pathlib import Path

from hydromt.model.components import ModelComponent

# from hydromt_sfincs.config_variables import SfincsConfigVariables
from hydromt_sfincs.config_variables import sfincs_config_variables

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel


class SfincsConfig(ModelComponent):
    """Class to read and write SFINCS input files."""

    def __init__(self, model: "SfincsModel"):
        self._filename = "sfincs.inp"
        self._data: sfincs_config_variables = None
        super().__init__(model=model)

    @property
    def data(self):
        """Return the SfincsConfig object."""
        if self._data is None:
            self._data = sfincs_config_variables
        return self._data

    def read(self, filename: str = "sfincs.inp") -> None:
        """Read a text file and populate SfincsConfig.
        This function also determines the grid type and updates the grid properties.
        """

        # Set the filename and check if it is an absolute path
        self._filename = filename
        if not isabs(filename):
            self._filename = join(self.root.path, filename)

        with open(self._filename, "r") as fid:
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

        # Convert dictionary to SfincsConfig instance
        self._data = self._data.copy(update=inp_dict)

        # Update the grid properties from the configuration
        # This will either drop the quadtree component or the regular component?
        self.update_grid_from_config()

    def write(self, filename: str = "sfincs.inp") -> None:
        """Write the instance's attributes to a file."""
        self.root._assert_write_mode
        if not isabs(filename) and self.root.path:
            self._filename = join(self.root.path, filename)

        with open(self._filename, "w") as fid:
            for key, value in self.data.dict(exclude_unset=False).items():
                if value is None:
                    continue
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

                if key in self.data.model_fields:
                    description = self.data.model_fields[key].description
                    if description:
                        # Add description to string
                        string = string.ljust(50) + f" # {description}"

                fid.write(string + "\n")

    def get(self, key: str, fallback: Any = None, abs_path: bool = False) -> Any:
        """Get a value with validation check."""

        value = self.data.model_dump().get(key, fallback)

        if value is None and fallback is not None:
            value = fallback
        if abs_path and isinstance(value, (str, Path)):
            value = Path(value)
            if not isabs(value):
                value = Path(abspath(join(self.root.path, value)))

        return value

    def set(self, key: str, value: Any, skip_validation=False) -> None:
        """Set a value with validation using Pydantic's model_copy."""

        if not hasattr(self.data, key):
            raise KeyError(f"'{key}' is not a valid attribute of SfincsConfig.")

        if not skip_validation:
            # Validate the new data
            # FIXME implement this in a better way
            # It works, but it is quite slow when all the variables are set in a loop
            # Therefore the skip_validation option is added
            try:
                self.data.model_validate({key: value})
            except Exception as e:
                raise TypeError(f"Invalid input type for '{key}'")

        self.data.__setattr__(key, value)

    def update(self, dict: Dict[str, Any]) -> None:
        """
        Update multiple attributes with validation from a dictionary with key-value pairs.

        Parameters:
        -----------
        dict (Dict[str, Any]):
            A dictionary containing key-value pairs to update the attributes.
            For example, dict = {'mmax': 100, 'nmax': 50}.
        """
        # set each key-value pair in the dictionary
        for key, value in dict.items():
            self.set(key, value)

    def update_grid_from_config(self) -> None:
        """Update the grid properties from the configuration."""

        # Determine grid type based on configuration
        self.model.grid_type = "quadtree" if self.get("qtrfile") else "regular"

        if self.model.grid_type == "regular":
            # update the regular grid properties from the configuration
            self.model.grid.update_grid_from_config()
            # drop quadtree component
            self.model.components.pop("quadtree", None)
            # TODO also drop mask and subgrid components?
        elif self.model.grid_type == "quadtree":
            # drop regular component
            self.model.components.pop("grid", None)

    def get_set_config_file_variable(
        self, key: str, value: str, default_filename: str
    ) -> Path:
        """Return filepath of a variable 'key', and add key-value to config if not present.

        Actions depending on situation:
         1) input file variable 'key' is given as input
            a) value is only the name of variable
                - add to config directly as value
                - return file_path including root
            b) value is a path
                - get file_directory and value_name through split
                - update the config
                    - with only value_name if file_directory==root
                    - otherwise with full path 'value'

         2) variable 'key' already in config:
             a) get full file_path using get with abs_path=True
                - In case not a path, then it adds the root

         3) use default name and root if not yet in config:
             a) set default name
             b) update the config
             c) give back full file_path

        Parameters:
        -----------
        key (str):
            Input filename like 'obsfile'
        value (str, Optional):
            Optional input filename corresponding 'obsfile',
            if not supplied, the default_filename will be used.
        default_filename (str):
            Default filename for corresponding 'key' like 'sfincs.obs'

        Returns:
        -----------
        file_path (Path):
            Full filename path of the file, as pathlib.Path
        """
        # Use pathlib.Path for modern, readable, and Pythonic code.

        # 1) input file variable 'key' is given as input
        if value is not None:
            # Split the file path
            file_directory, value_name = split(value)

            if file_directory == "":  # dealing with only a file name as input
                # add to config directly as value:
                self.model.config.set(key, value)
                # return file_path including root:
                file_path = Path(abspath(join(self.model.config.root.path, value)))

            else:  # dealing with a path as input
                # check if path == root, determines how we add to the config:
                if Path(abspath(file_directory)) == self.model.config.root.path:
                    # folders are the same, only write value_name:
                    self.model.config.set(key, value_name)
                else:
                    # file_directory different than root

                    # add the full path to config:
                    self.model.config.set(key, value)

                # value is the full_path already:
                file_path = Path(value)

        # 2) variable 'key' already in config:
        elif value is None and self.model.config.get(key) is not None:
            # If variable 'key' is not None, it has been set already in config
            # NOTE Assumes that by default all input file variables in
            # SfincsInputVariables are initiated as None

            # get existing file name as full path (already adds root, in case not a file);
            file_path = self.model.config.get(
                key, abs_path=True
            )  # return is 'Path' directly

        # 3) use default name and root:
        elif value is None and self.model.config.get(key) is None:
            # If variable 'key' is None, it has not been added previously to config
            # Now add the default_filename to config

            self.model.config.set(key, default_filename)

            # And return the full file_path including root
            file_path = Path(
                abspath(join(self.model.config.root.path, default_filename))
            )

        return file_path

    def get_set_file_variable(
        self, key: str, value: str | Path = None, default: str = None
    ) -> Path:
        """Return filepath of a variable 'key', and add key-value to config if not present.

        Parameters:
        -----------
        key (str):
            Input keyword, e.g. "obsfile"
        value (str, Optional):
            Optional input filename, e.g. "sfincs.obs"
        default (str, Optional):
            Default filename for corresponding 'key', e.g. 'sfincs.obs'

        When getting the input path to READ the file, default should NOT be provided.
        When setting the output path to WRITE the file, default SHOULD be provided.

        This method does NOT check if the file exists or not.

        Returns:
        -----------
        file_path (Path):
            Full filename path of the file, as pathlib.Path
        """

        # If value is a string, turn it into a Path
        if isinstance(value, str):
            value = Path(value)

        root_path = self.model.root.path.resolve()

        if value is not None:
            # File name is given as input
            file_directory = value.parent
            file_name = value.name
            if file_directory == ".":
                # Dealing with only a file name as input
                # Add to config directly
                self.model.config.set(key, file_name)
                # Return file path including root
                full_file_path = root_path / file_name

            else:
                # Dealing with a path as input
                # Check if path is same as root
                if Path(file_directory).resolve() == root_path:
                    # Folders are the same, only write file_name
                    self.model.config.set(key, file_name)
                    full_file_path = root_path / file_name
                else:
                    # File directory different than root
                    # Check if file_directory is an absolute or relative path
                    if Path(file_directory).is_absolute():
                        # Add the full path to config
                        self.model.config.set(key, str(value))
                        full_file_path = value
                    else:
                        # Relative path
                        self.model.config.set(key, str(value))
                        full_file_path = (root_path / value).resolve()

        else:  # Input file name not provided so get it from the config
            # Get existing file name from config
            value = self.get(key)
            if value is None:
                if default is None:
                    # File name not defined in config, so return None
                    return None
                else:
                    # Default file name is provided
                    self.model.config.set(key, default)
                    full_file_path = root_path / default
            else:
                value = Path(value)
                file_directory = value.parent
                file_name = value.name
                if file_directory == ".":
                    # Dealing with only a file name as input
                    full_file_path = root_path / file_name
                else:
                    # Dealing with a path as input
                    if Path(file_directory).resolve() == root_path:
                        # Folders are the same, only write file_name
                        full_file_path = root_path / file_name
                    else:
                        # File directory different than root
                        # Check if file_directory is an absolute or relative path
                        if Path(file_directory).is_absolute():
                            # Absolute
                            full_file_path = value
                        else:
                            # Relative
                            full_file_path = (root_path / value).resolve()

        return full_file_path
