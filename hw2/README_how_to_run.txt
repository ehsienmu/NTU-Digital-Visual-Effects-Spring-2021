[拍攝了兩組場景]

我們拍了兩個景，第二個景的data在data/scene2

---------------------------------------------------------

[focal length檔案格式]

可以是
1. autostitch產生的pano.txt格式，
2. 也可以是：
image00.jpg
400.235
image01.jpg
401.369

範例請看data內中的pano.txt（格式1.）或clean_pano.txt（格式2.）

---------------------------------------------------------

[執行參數]

我們使用了Argparser來執行程式：

usage: proj2.py [-h] [-photo_path PHOTO_PATH] [-focal_filename FOCAL_FILENAME]
                [-detect_threshold DETECT_THRESHOLD]
                [-match_threshold MATCH_THRESHOLD]

optional arguments:（都有預設值）
  -h, --help                           show this help message and exit
  -photo_path PHOTO_PATH               photo path                       , (default: 'cwd')
  -focal_filename FOCAL_FILENAME       focal_length.txt filename        , (default: 'pano.txt') , 可輸入絕對路徑或非絕對路徑。若不是輸入絕對路徑，如pano.txt，則會認定是photo_path + pano.txt
  -detect_threshold DETECT_THRESHOLD   detect feature threshold         , (default: '50000')    , 調大一點feature比較少會跑比較快
  -match_threshold MATCH_THRESHOLD     match threshold                  , (default: '400')

---------------------------------------------------------

[執行範例]

example:
> python proj2.py -photo_path D:/109-2/digivfx/hw2/data -focal_filename pano.txt
> python proj2.py -photo_path D:/109-2/digivfx/hw2/data -focal_filename D:\109-2\digivfx\hw2\data\clean_pano.txt
> python proj2.py -photo_path D:/109-2/digivfx/hw2/data -focal_filename clean_pano.txt
如果focal_length檔名叫做pano.txt，則可:
> python proj2.py -photo_path D:/109-2/digivfx/hw2/data