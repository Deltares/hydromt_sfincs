import logging
from os.path import abspath, isabs, join
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from hydromt import hydromt_step
from hydromt.model.components import ModelComponent

from hydromt_sfincs.components.config.config_variables import SfincsConfigVariables

if TYPE_CHECKING:
    from hydromt_sfincs import SfincsModel

logger = logging.getLogger(f"hydromt.{__name__}")


class SfincsConfig(ModelComponent):
    """ " Class to read and write SFINCS configuration files (sfincs.inp).

    This class is the interface between the model and its
    ([`SfincsConfigVariables`](hydromt_sfincs.components.config.config.SfincsConfigVariables))
    data attribute, which performs the actual reading, writing and validation of
    configuration variables. It manages file paths and the model's grid/root state.

    See Also
    --------
    :py:class:`~hydromt_sfincs.components.config.SfincsConfigVariables`
        Pydantic model class for SFINCS configuration variables.
    """

    def __init__(self, model: "SfincsModel"):
        self._filename = "sfincs.inp"
        self._data: SfincsConfigVariables = None
        self._read_root: Path = None
        super().__init__(model=model)
        # Lock the read root and filename at init time so that lazy reads
        # triggered after a root change still find the original files.
        if self.root.is_reading_mode():
            self._read_root = model.root.path.resolve()
            self._filename = self._read_root / "sfincs.inp"

    @property
    def data(self):
        """Return the Pydantic SfincsConfigVariables object."""
        if self._data is None:
            self._data = SfincsConfigVariables()
            if self.root.is_reading_mode():
                self.read()
        return self._data

    @property
    def filename(self) -> Path:
        """Return the absolute filename of the SFINCS input file."""
        if not Path(self._filename).is_absolute():
            self._filename = self.model.root.path.resolve() / "sfincs.inp"
        return Path(self._filename)

    def read(self, update_changed_field_names: bool = False) -> None:
        """Read a text file with the sfincs configuration from the root folder and populate
        the SfincsConfigVariables. This function also determines the grid type and updates
        the grid properties of the SfincsModel (e.g. crs and extent).

        Parameters:
        -----------
        update_changed_field_names (bool):
            If True, migrate legacy configuration keys to their current names.
            Default is False.
        """

        self.root._assert_read_mode()

        self._data = SfincsConfigVariables.read(
            self.filename,
            update_changed_field_names=update_changed_field_names,
        )

        # Update the grid properties from the configuration
        # This will either drop the quadtree component or the regular component?
        self.update_grid_from_config()

    def write(
        self,
        filename: str = "sfincs.inp",
        write_description: bool = False,
        write_comments: bool = False,
        explicit_only: bool = False,
    ) -> None:
        """Write the SfincsConfigVariables to a text file in the root folder of the model.

        Parameters:
        -----------
        filename (str):
            The name of the file to write the configuration to. Default is "sfincs.inp".
        write_description (bool):
            If True, append the schema field description as a trailing comment.
            Default is False.
        write_comments (bool):
            If True, append the original inline comment read from the source
            file (if any) as a trailing comment, taking priority over
            write_description for keys that have one. Default is False.
        explicit_only (bool):
            If True, write only fields explicitly read or set. Default is False.
        """
        self.root._assert_write_mode()

        if not isabs(filename) and self.root.path:
            self._filename = self.root.path / filename

        self.data.write(
            self.filename,
            write_description=write_description,
            write_comments=write_comments,
            explicit_only=explicit_only,
        )

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

        value = getattr(self.data, key, fallback)

        if value is None and fallback is not None:
            value = fallback
        if abs_path and isinstance(value, (str, Path)):
            value = Path(value)
            if not isabs(value):
                # Use the root that was active when the config was read so that
                # a later root change (e.g. cloning the model) does not redirect
                # reads to the new, empty root.  Fall back to the current root
                # when no read root is recorded (write-only mode) or when the
                # caller did not supply a fallback (write context).
                read_root = getattr(self, "_read_root", None)
                if read_root is not None and fallback is not None:
                    value = (read_root / value).resolve()
                else:
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
        self._data = self.data.set_value(key, value, skip_validation=skip_validation)

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
            self._data = self.data.set_values(updates, skip_validation=skip_validation)

    def update_grid_from_config(self) -> None:
        """Update the grid properties from the configuration. This method determines the grid type
        based on the presence of the 'qtrfile' variable in the configuration. If 'qtrfile' is set,
        the grid type is set to 'quadtree'; otherwise, it is set to 'regular'.
        """

        # Determine grid type based on configuration
        self.model._grid_type = "quadtree" if self.get("qtrfile") else "regular"

        if self.model.grid_type == "regular":
            # update the regular grid properties from the configuration
            self.model.grid.update_grid_from_config()

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

        # Make sure the value is an absolute path.
        # When the caller did not supply an explicit value or a write-mode
        # default, the path came from the original sfincs.inp and should be
        # resolved against the root that was active at read time so that a
        # later root change does not redirect reads to the wrong directory.
        if not value_path.is_absolute():
            read_root = getattr(self, "_read_root", None)
            if read_root is not None and value is None and default is None:
                return (read_root / value_path).resolve()
            return (root_path / value_path).resolve()
        else:
            return value_path
