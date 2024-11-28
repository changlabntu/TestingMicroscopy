import glob
import subprocess
import tifffile as tiff

prj_list = sorted(glob.glob('/media/ExtHDD01/logs/DPM4X/ae/iso0*'))

for prj in prj_list[3:]:
    epoch_list = sorted(glob.glob(prj + '/checkpoints/encoder*'))
    epoch_list = [x.split('_')[-1].split('.')[0] for x in epoch_list]

    prj = prj.split('/')[-1]
    for e in epoch_list[1::4]:
        print(prj, e)
        cmd = [
            "python",
            "test_combine.py",
            "--prj", '/ae/' + prj + '/',
            "--epoch", e,
            "--model_type", "AE",
            "--option", "DPM4X",
            "--gpu",
            "--hbranchz",
            "--reverselog"
        ]

        process = subprocess.run(cmd, capture_output=True, text=True)
        # Check return code
        if process.returncode == 0:
            print("Command executed successfully")
            print("Output:", process.stdout)
        else:
            print("Error occurred")
            print("Error output:", process.stderr)

        xy = tiff.imread('/media/ExtHDD01/Dataset/paired_images/DPM4X/xy.tif')
        tiff.imwrite('/media/ExtHDD01/Dataset/paired_images/DPM4X/imgout/' + prj + '_' + e + '.tif', xy[122,:,:])


