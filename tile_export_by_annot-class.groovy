/*
@Cunyuan JI
Run for project!

This script exports tiles from annotated WSIs in QuPath, only the annotation areas are considered
Notice that the tiles belonging to different specimen classes will be stored in different directories

注意： 对整个项目运行。

本文件从 QUPath 导出 tile 。只处理标注的部分
对标为不同类别的区域，其 tile 将被存储到不同的目录中

比如，在标注被打上"healthy tissue", "tumor", "blood"， "other tissues"几种tag的场合，
这段代码得到的目录是这样的：

Tree:

Project
    --DATA
        -- 01_17... （片子编号）
            -- IHC （你的染色法）
                -- Annotations （qupath标注信息文件）
                    -- txt file
                -- Tiles （你的图像网格数据）
                    -- Tumor （类别）
                        -- ROI_1...
                        -- ROI_2...
                    -- Blood （类别）
                    -- Other Tissues （类别）
            -- HE 
                -- Tumor
                -- Blood
                -- Other Tissues
 
 */

import qupath.lib.objects.PathCellObject
import qupath.lib.objects.PathDetectionObject
import qupath.lib.objects.PathObject
import qupath.lib.objects.PathObjects
import qupath.lib.objects.PathTileObject
import qupath.lib.regions.RegionRequest
import qupath.lib.roi.RoiTools
import qupath.lib.roi.interfaces.ROI

import java.awt.geom.AffineTransform

import static qupath.lib.gui.scripting.QPEx.*
import groovy.util.FileTreeBuilder

def server = getCurrentServer()
def downsample = 1.0000000

// 改成你的项目路径
def ProjBaseDir = "/Users/cunyuan/DATA/Kimura/qupath-proj/"


/*
For WSIs:
    write annotation data to text file
    For annotations:
        For classes:
            Make directory in /Tiles
            For tiles:
                Write to directory
 */

// ！！！
// ROI 边缘扩大的相对系尺寸。假设为0.5，那么左右多出宽为原图一半的边缘，图片尺寸扩大两倍
marginsize = 0.5
// 改成你的染色类型，和片子的文件名一致
def stain_types = ["HE", "IHC"]
// Process all HE images in the project.
def ImgName = GeneralTools.getNameWithoutExtension(server.getMetadata().getName())

for (stain_name in stain_types) {
    if (ImgName.contains(stain_name)) {
        WSI_ID = ImgName
    } else {
        continue
    }

// Get the project & the requested image name

    def Objects = getAnnotationObjects()

//        new File(ProjBaseDir + "${stain_name}/${WSI_ID}").mkdirs()
    new File(ProjBaseDir + "${stain_name}/${WSI_ID}/Annotations/").mkdirs()
    def annotations = getAnnotationObjects().collect { new qupath.lib.objects.PathAnnotationObject(it.getROI(), it.getPathClass()) }
    print(annotations)
    new File(buildFilePath(ProjBaseDir + "${stain_name}/${WSI_ID}/Annotations/annotations-" + ImgName + '.txt')).withObjectOutputStream {
        it.writeObject(annotations)
    }
    // Write tiles
    def o = 0
    for (pathObject in Objects) {
        print(pathObject)
        if (pathObject.hasChildren()) {
            className = pathObject.getProperties().get('displayedName')
            print(className)
            new File(ProjBaseDir + "${stain_name}/${WSI_ID}/Tiles/${className}").mkdirs()
            //export tiles in the dir
            tiles = pathObject.getChildObjects()
            def k = 0
            for (tile in tiles) {
                roi = tile.getROI()
                
                def sizey = roi.getBoundsHeight()
                def sizex = roi.getBoundsWidth()
                def sx = roi.getBoundsX() 
                // - (int)(sizex * marginsize)
                def sy = roi.getBoundsY() 
                // - (int)(sizey * marginsize) //比如margin=0.5就是向左上左下偏移半个格子
                // sizex += marginsize*2 
                // sizey += marginsize*2  // 比如margin=0.5就把tile大小扩大一倍

                // ！！！
                // 把扩大ROI的操作写成下面的仿射变换
                // def matrix = [1 + marginsize*2 , 0.0000, -(sx + sizex * marginsize),
                //              0.0000, 1+ marginsize*2 , -(sy + sizey * marginsize)]
                // // 应用仿射变换
                // def transform = new AffineTransform(
                //             matrix[0] as double, matrix[3] as double, matrix[1] as double,
                //             matrix[4] as double, matrix[2] as double, matrix[5] as double
                //     )
                // def roi2 = transformROI(roi, transform)
                
                requestedTile = RegionRequest.createInstance(server.getPath(), downsample, roi)
                // 格式化输出文件名。请按需修改
                // 原始ROI（无重叠）
                name = "${ProjBaseDir}${stain_name}/${WSI_ID}/Tiles/${className}/${o}_${k}_${WSI_ID}_(d=$downsample, x=$sx, y=$sy, w=$sizex, h=$sizey, z=${k})_origROI.tif".toString()
                print(roi)
                writeImageRegion(server, requestedTile, name) //保存

                // // 扩大ROI （有重叠）
                // requestedTile2 = RegionRequest.createInstance(server.getPath(), downsample, roi2)
                // name = "${ProjBaseDir}${stain_name}/${WSI_ID}/Tiles/${className}/${o}_${k}_${WSI_ID}_(d=$downsample, x=$sx, y=$sy, w=$sizex, h=$sizey, z=${k})_extendedROI.tif".toString()
                // print(roi2)
                // writeImageRegion(server, requestedTile2, name) // 保存
                k += 1
                // 调试用
            //    if (k > 5) {
            //        break
            //    }
            }
        }
        o += 1
    }
}

// ROI变换函数
ROI transformROI(ROI roi, AffineTransform transform) {
    def shape = RoiTools.getShape(roi)
    // Should be able to use roi.getShape() - but there's currently a bug in it for rectangles/ellipses!
    shape2 = transform.createTransformedShape(shape)
    return RoiTools.getShapeROI(shape2, roi.getImagePlane(), 0.5)
}
