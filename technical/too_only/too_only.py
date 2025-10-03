__all__ = (
    "generate_baseline_coresched",
    "example_scheduler",
    "sched_argparser",
    "set_run_info",
    "run_sched",
    "gen_long_gaps_survey",
    "gen_greedy_surveys",
    "generate_blobs",
    "generate_twi_blobs",
    "generate_twilight_near_sun",
    "standard_bf",
)

import argparse
import os
import subprocess
import sys

import healpy as hp
import numpy as np
import rubin_scheduler
import rubin_scheduler.scheduler.basis_functions as bf
import rubin_scheduler.scheduler.detailers as detailers
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.utils import iers
from rubin_scheduler.scheduler import sim_runner
from rubin_scheduler.scheduler.model_observatory import ModelObservatory, tma_movement
from rubin_scheduler.scheduler.schedulers import CoreScheduler, SimpleBandSched
from rubin_scheduler.scheduler.surveys import (
    BlobSurvey,
    GreedySurvey,
    LongGapSurvey,
    ScriptedSurvey,
    gen_roman_off_season,
    gen_roman_on_season,
    gen_too_surveys,
)
from rubin_scheduler.scheduler.targetofo import gen_all_events
from rubin_scheduler.scheduler.utils import (
    ConstantFootprint,
    CurrentAreaMap,
    make_rolling_footprints,
)
from rubin_scheduler.site_models import Almanac
from rubin_scheduler.utils import DEFAULT_NSIDE, SURVEY_START_MJD, _hpid2_ra_dec

#from ddf_df_gen import generate_ddf_df
#from ddf_presched import generate_ddf_scheduled_obs

STANDARD_EXP_TIME = 30.0
STANDARD_EXP_TIME_u = 38.0

# So things don't fail on hyak
iers.conf.auto_download = False
# XXX--note this line probably shouldn't be in production
iers.conf.auto_max_age = None


def example_scheduler(**kwargs):
    """Renamed"""
    return generate_baseline_coresched(**kwargs)


def generate_baseline_coresched(
    nside: int = DEFAULT_NSIDE,
    mjd_start: float = SURVEY_START_MJD,
    no_too: bool = False,
) -> CoreScheduler:
    """Provide an example baseline survey-strategy scheduler.

    Parameters
    ----------
    nside : `int`
        Nside for the scheduler maps and basis functions.
    mjd_start : `float`
        Start date for the survey (MJD).
    no_too : `bool`
        Turn off ToO simulation. Default False.

    Returns
    -------
    scheduler : `rubin_scheduler.scheduler.CoreScheduler`
        A scheduler set up as the baseline survey strategy.
    """
    parser = sched_argparser()
    args = parser.parse_args(args=[])
    args.setup_only = True
    args.no_too = no_too
    args.dbroot = "example_"
    args.outDir = "."
    args.nside = nside
    args.mjd_start = mjd_start
    scheduler = gen_scheduler(args)
    return scheduler




def set_run_info(dbroot=None, file_end="", out_dir="."):
    """Gather versions of software used to record"""
    extra_info = {}
    exec_command = ""
    for arg in sys.argv:
        exec_command += " " + arg
    extra_info["exec command"] = exec_command
    try:
        extra_info["git hash"] = subprocess.check_output(["git", "rev-parse", "HEAD"])
    except subprocess.CalledProcessError:
        extra_info["git hash"] = "Not in git repo"

    extra_info["file executed"] = os.path.realpath(__file__)
    try:
        rs_path = rubin_scheduler.__path__[0]
        hash_file = os.path.join(rs_path, "../", ".git/refs/heads/main")
        extra_info["rubin_scheduler git hash"] = subprocess.check_output(
            ["cat", hash_file]
        )
    except subprocess.CalledProcessError:
        pass

    # Use the filename of the script to name the output database
    if dbroot is None:
        fileroot = os.path.basename(sys.argv[0]).replace(".py", "") + "_"
    else:
        fileroot = dbroot + "_"
    fileroot = os.path.join(out_dir, fileroot + file_end)
    return fileroot, extra_info


def run_sched(
    scheduler,
    survey_length=365.25,
    nside=DEFAULT_NSIDE,
    filename=None,
    verbose=False,
    extra_info=None,
    illum_limit=40.0,
    mjd_start=60796.0,
    event_table=None,
    sim_to_o=None,
    snapshot_dir=None,
    readtime=3.07,
    band_changetime=140.0,
    tma_performance=40,
):
    """Run survey"""
    n_visit_limit = None
    fs = SimpleBandSched(illum_limit=illum_limit)
    observatory = ModelObservatory(nside=nside, mjd_start=mjd_start, sim_to_o=sim_to_o)

    tma_kwargs = tma_movement(percent=tma_performance)
    observatory.setup_telescope(**tma_kwargs)
    observatory.setup_camera(band_changetime=band_changetime, readtime=readtime)

    observatory, scheduler, observations = sim_runner(
        observatory,
        scheduler,
        sim_duration=survey_length,
        filename=filename,
        delete_past=True,
        n_visit_limit=n_visit_limit,
        verbose=verbose,
        extra_info=extra_info,
        band_scheduler=fs,
        event_table=event_table,
        snapshot_dir=snapshot_dir,
        allow_deadtime=True,
        skip_to_events=True,
    )

    return observatory, scheduler, observations


def gen_scheduler(args):
    survey_length = args.survey_length  # Days
    out_dir = args.out_dir
    verbose = args.verbose
    nexp = args.nexp
    dbroot = args.dbroot
    nside = args.nside
    mjd_plus = args.mjd_plus
    split_long = args.split_long
    snapshot_dir = args.snapshot_dir

    # Parameters that were previously command-line
    # arguments.
    illum_limit = 40.0  # Percent. Lunar illumination used for band loading
    u_exptime = 38.0  # Seconds
    nslice = 2  # N slices for rolling
    rolling_scale = 0.9  # Strength of rolling
    rolling_uniform = True  # Should we use the uniform rolling flag
    nights_off = 3  # For long gaps
    ei_night_pattern = 4  # select doing earth interior observation every 4 nights
    ei_bands = "riz"  # Bands to use for earth interior observations.
    ei_repeat = 4  # Number of times to repeat earth interior observations
    ei_am = 2.5  # Earth interior airmass limit
    ei_elong_req = 45.0  # Solar elongation required for inner solar system
    ei_area_req = 0.0  # Sky area required before attempting inner solar system
    per_night = True  # Dither DDF per night
    camera_ddf_rot_limit = 75.0  # degrees

    # Be sure to also update and regenerate DDF grid save file
    # if changing mjd_start
    mjd_start = SURVEY_START_MJD + mjd_plus

    fileroot, extra_info = set_run_info(
        dbroot=dbroot,
        file_end="v5.0.1_",
        out_dir=out_dir,
    )

    pattern_dict = {
        1: [True],
        2: [True, False],
        3: [True, False, False],
        4: [True, False, False, False],
        # 4 on, 4 off
        5: [True, True, True, True, False, False, False, False],
        # 3 on 4 off
        6: [True, True, True, False, False, False, False],
        7: [True, True, False, False, False, False],
    }
    ei_night_pattern = pattern_dict[ei_night_pattern]

    sky = CurrentAreaMap(nside=nside)
    footprints_hp_array, labels = sky.return_maps()

    wfd_indx = np.where((labels == "lowdust") | (labels == "virgo"))[0]
    wfd_footprint = footprints_hp_array["r"] * 0
    wfd_footprint[wfd_indx] = 1

    footprints_hp = {}
    for key in footprints_hp_array.dtype.names:
        footprints_hp[key] = footprints_hp_array[key]

    footprint_mask = footprints_hp["r"] * 0
    footprint_mask[np.where(footprints_hp["r"] > 0)] = 1

    too_scale = 1.0
    sim_ToOs, event_table = gen_all_events(scale=too_scale, nside=nside)
    camera_rot_limits = [-80.0, 80.0]
    detailer_list = []
    detailer_list.append(
        detailers.CameraRotDetailer(
            min_rot=np.min(camera_rot_limits), max_rot=np.max(camera_rot_limits)
        )
    )
    detailer_list.append(detailers.LabelRegionsAndDDFs())
    # Let's make a footprint to follow up ToO events
    too_footprint = footprints_hp["r"] * 0 + np.nan
    too_footprint[np.where(footprints_hp["r"] > 0)[0]] = 1.0

    toos = gen_too_surveys(
        nside=nside,
        detailer_list=detailer_list,
        too_footprint=too_footprint,
        split_long=split_long,
        n_snaps=nexp,
    )
    surveys = [toos]

    scheduler = CoreScheduler(surveys, nside=nside)

    if args.setup_only:
        return scheduler
    else:
        years = np.round(survey_length / 365.25)
        observatory, scheduler, observations = run_sched(
            scheduler,
            survey_length=survey_length,
            verbose=verbose,
            filename=os.path.join(fileroot + "%iyrs.db" % years),
            extra_info=extra_info,
            nside=nside,
            illum_limit=illum_limit,
            mjd_start=mjd_start,
            event_table=event_table,
            sim_to_o=sim_ToOs,
            snapshot_dir=snapshot_dir,
        )
        return observatory, scheduler, observations


def sched_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose", dest="verbose", action="store_true", help="Print more output"
    )
    parser.set_defaults(verbose=False)
    parser.add_argument(
        "--survey_length", type=float, default=365.25 * 10, help="Survey length in days"
    )
    parser.add_argument("--out_dir", type=str, default="", help="Output directory")
    parser.add_argument(
        "--nexp", type=int, default=1, help="Number of exposures per visit"
    )
    parser.add_argument("--dbroot", type=str, help="Database root")
    parser.add_argument(
        "--setup_only",
        dest="setup_only",
        default=False,
        action="store_true",
        help="Only construct scheduler, do not simulate",
    )
    parser.add_argument(
        "--nside",
        type=int,
        default=DEFAULT_NSIDE,
        help="Nside should be set to default (32) except for tests.",
    )
    parser.add_argument(
        "--mjd_plus",
        type=float,
        default=0,
        help="number of days to add to the mjd start",
    )
    parser.add_argument(
        "--split_long",
        dest="split_long",
        action="store_true",
        help="Split long ToO exposures into standard visit lengths",
    )
    parser.add_argument(
        "--snapshot_dir",
        type=str,
        default="",
        help="Directory for scheduler snapshots.",
    )
    parser.set_defaults(split_long=False)
    parser.add_argument("--no_too", dest="no_too", action="store_true")
    parser.set_defaults(no_too=False)

    return parser


if __name__ == "__main__":
    parser = sched_argparser()
    args = parser.parse_args()
    gen_scheduler(args)
