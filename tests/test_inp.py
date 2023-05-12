from hydromt_sfincs.sfincs_input import SfincsInput


def test_inp(tmpdir):
    inpdict = {
        "mmax": 84,
        "nmax": 36,
        "dx": 150,
        "dy": 150,
        "x0": 318650.0,
        "y0": 5034000.0,
        "rotation": 27.0,
        "epsg": 32633,
    }
    inp = SfincsInput.from_dict(inpdict)
    assert all([inpdict[k] == inp[k] for k in inpdict])
    # to dict
    inpdict1 = inp.to_dict()
    assert all([inpdict[k] == inpdict1[k] for k in inpdict])
    # test __get__ and __set__
    inp["mmax"] = 42
    assert inp["mmax"] == 42
    # write and read
    inp0 = SfincsInput()  # initialize with default values
    fn_out = str(tmpdir.join("sfincs.inp"))
    inp0.write(fn_out)
    inp1 = SfincsInput.from_file(fn_out)
    assert inp0 == inp1
    # print
    assert str(inp0).startswith("SfincsInput(")
