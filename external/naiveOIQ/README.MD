
# naiveOIQ
> naive_OutImageQuality

这是一个可以计算并输出图片质量的工具




 - |原图 ![原图](1.jpg)| 模糊 ![模糊](1_模糊.jpg)
---|---|----
`clarity` 清晰度|6.78223|2.76616
`blurriness` 模糊度|0.433176|2.76616
`entropy` 熵|6.4703|6.49129
`PSNR`  | |30.195
`SSIM`  | |0.944815




- |20 质量.jpg ![20质量](1_20质量.jpg)|40 质量.jpg ![40质量](1_40质量.jpg)
---|---|----
`clarity` 清晰度|5.65042|6.29638
`blurriness` 模糊度|0.512679|0.468921
`entropy` 熵|5.51854|6.53426
`PSNR` 峰值信噪比|30.0938|33.2334
`SSIM` 结构相似度|0.907538|0.958402


## 功能

输出图片质量信息：  
`naiveOIQ image `

输出图片质信息，并且与另一图片比较，计算差异：   
`naiveOIQ image OriginalImage`

加入 `-q`参数快速比较，但是 SSIM 信息会没那么准确   
`naiveOIQ image OriginalImage -q`

加入 `-a`会得到 G、B、R 每个通道各自的 SSIM 比较信息，速度慢   
`naiveOIQ image OriginalImage -a`


### `clarity` 清晰度
图片的清晰度，越高清晰度越高。

### `blurriness` 模糊度
图片的模糊程度，越高说明图像越模糊。

### `entropy` 熵
图片的信息混乱程度，一般压缩会让熵值降低。

### `PSNR` c
>PSNR 在这指峰值信噪比

与原图片越相似, PSNR 越高。
画质损失会让 PSNR 降低。

### `SSIM` 平均结构相似度
>这里 SSIM 是指 MSSIM（平均结构相似度）

平均结构相似度越接近 1 图片越相似。
画质损失会让 SSIM 降低。
