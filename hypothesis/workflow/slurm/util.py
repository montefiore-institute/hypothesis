import os
import shutil


def slurm_detected():
    output = shutil.which("squeue")

    return output != None and len(shutil.which("squeue")) > 0
