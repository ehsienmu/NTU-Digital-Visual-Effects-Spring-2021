

我們拍了兩個景，第二個景的data在data/場景2

---------------------------------------------------------

我們使用了Argparser來執行程式：

usage: hw1.py [-h] [-r RADIANCE_METHOD] [-t TONE_MAPPING_METHOD] [-mul MUL]
              [-a A] [-sigma SIGMA] [-e EPS] [-phi PHI] [-w L_WHITE]
              input_image_file

positional arguments:
  input_image_file      Images directory path        , you can type 'cwd' to get pathname of current working directory

optional arguments:（都有預設值）
  -r                    Radiance method              , 'Debevec'(default), 'Robertson'
  -t                    Tone mapping method method   , 'Drago', 'Global'(default), 'Local'
  -mul                  mul parameter in Drago       , (default: 1.2)
  -a                    a parameter in Global/Local  , (default: 1.0)
  -sigma                sigma parameter in Local     , (default: 0.000001)
  -e                    epsilon parameter in Local   , (default: 0.004)
  -phi                  phi parameter in Local       , (default: 8)
  -w                    l_white in Global            , (default: 1.5)




example:
> python hw1.py D://109-2//digivfx//hw1_[14]//data
