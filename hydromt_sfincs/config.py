from datetime import datetime
from typing import TYPE_CHECKING, List, Optional, Dict, Any
from ast import literal_eval
from os.path import abspath, isabs, join
from pathlib import Path

from hydromt.model.components import ModelComponent

from hydromt_sfincs.config_variables import SfincsConfigVariables

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel


class SfincsConfig(ModelComponent):
    """Class to read and write SFINCS input files."""

    def __init__(self, model: "SfincsModel"):
        self._filename = "sfincs.inp"
        self._data: SfincsConfigVariables = None
        super().__init__(model=model)

    @property
    def data(self) -> SfincsConfigVariables:
        """Return the SfincsConfig object."""
        if self._data is None:
            self._data = SfincsConfigVariables()
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
        self._data = SfincsConfigVariables(**inp_dict)

        # Update the grid properties from the configuration
        self.update_grid_from_config()

    def write(self, filename: str = "sfincs.inp") -> None:
        """Write the instance's attributes to a file."""
        self.root._assert_write_mode
        if not isabs(filename) and self.root.path:
            self._filename = join(self.root.path, filename)

        with open(self._filename, "w") as fid:
            for key, value in self.data.dict(exclude_unset=True).items():
                if value is None:
                    continue
                if isinstance(value, float):  # remove insignificant traling zeros
                    string = f"{key.ljust(20)} = {value}\n"
                elif isinstance(value, int):
                    string = f"{key.ljust(20)} = {value}\n"
                elif isinstance(value, list):
                    valstr = " ".join([str(v) for v in value])
                    string = f"{key.ljust(20)} = {valstr}\n"
                elif hasattr(value, "strftime"):
                    dstr = value.strftime("%Y%m%d %H%M%S")
                    string = f"{key.ljust(20)} = {dstr}\n"
                else:
                    string = f"{key.ljust(20)} = {value}\n"
                fid.write(string)

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

    def set(self, key: str, value: Any) -> None:
        """Set a value with validation using Pydantic's model_copy."""
        if not hasattr(self.data, key):
            raise KeyError(f"'{key}' is not a valid attribute of SfincsConfig.")

        # Validate the new data
        # FIXME implement this in a better way
        try:
            value = SfincsConfigVariables(**{key: value}).__dict__[key]
        except Exception as e:
            raise TypeError(f"Invalid input type for '{key}'")

        self._data = self._data.model_copy(update={key: value})

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
        elif self.model.grid_type == "quadtree":
            # drop regular component
            self.model.components.pop("grid", None)
