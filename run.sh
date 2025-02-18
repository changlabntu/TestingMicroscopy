python test_only.py --gpu --config dpmfull2R --save ori xy --augmentation decode --fp16 --testcube --option VMAT
python test_assemble.py  --config dpmfull2R  --targets xy ori --option VMAT --image_datatype uint8
python test_only.py --gpu --config dpmfull2R --save ori xy --augmentation decode --fp16 --testcube --option DPM
python test_assemble.py  --config dpmfull2R --targets xy ori --option DPM --image_datatype uint8
