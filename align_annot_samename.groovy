/**
 * Script to transfer QuPath objects from one image to another, applying an AffineTransform to any ROIs.
 Get the transform matrix from qupath interactive alignment

 注意：只要运行一次即可，无需“对整个项目运行”。

 本脚本将 Qupath 的多边形形标注做批量变换，用来原封不动地复制HE和IHC染色图像上的标注对象
 比如：在H&E片子上做了标注，那么：
 1. 先计算H&E和IHC之间的变换
 2. 修改运行此脚本，即可将H&E上的标注变换到IHC上的正确位置
 这样，我们得到的标注在空间位置上是对齐的。之后再运行tile导出脚本即可
 More info on custom scripting: 修改自：
 https://gist.github.com/Svidro/5829ba53f927e79bb6e370a6a6747cfd

 # @ JI Cunyuan
 */

import qupath.lib.objects.PathCellObject
import qupath.lib.objects.PathDetectionObject
import qupath.lib.objects.PathObject
import qupath.lib.objects.PathObjects
import qupath.lib.objects.PathTileObject
import qupath.lib.roi.RoiTools
import qupath.lib.roi.interfaces.ROI

import java.awt.geom.AffineTransform

import static qupath.lib.gui.scripting.QPEx.*


def ihc_entrys = project.getImageList()

// Process all HE images in the project.
k = 0
for (ihc_entry in ihc_entrys[0..-1]) {
    // if img contains annot
    //      find entry1 with the same __name_
    //          if entry1 contains no annot
    //              move annot in img to entry1
    ihcHierarchy = ihc_entry.readHierarchy()
    ihcObjects = ihcHierarchy.getAnnotationObjects()

    for (ihc_pair in ihc_entrys[k+1..-1]) {
        if (ihc_pair.getImageName() == ihc_entry.getImageName()){
            pairData = ihc_pair.readImageData()
            ihcHierarchy1 = pairData.getHierarchy()
            ihcObjects1 = ihcHierarchy1.getAnnotationObjects()
            print("${ihcObjects}, ${ihcObjects1}")
            n0 = ihc_entry.getImageName()
            n1 = ihc_pair.getImageName()
            print("${n0}, ${n1}")
            if(ihcObjects1==[]){
                ihcHierarchy1.addPathObjects(ihcObjects)
                removeObjects(ihcHierarchy1.getDetectionObjects(), true)
                ihc_pair.saveImageData(pairData)
                print(n1 + ' processed.')
            
            }
        
        // ihcHierarchy1.clearAll()
        // ihc_pair.saveImageData(pairData)

        }

    }


    k = k+1
}

print 'Done!'
