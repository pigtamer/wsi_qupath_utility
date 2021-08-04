/*
@Cunyuan JI
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
def ProjBaseDir = '/Users/cunyuan/DATA/'


// ！！！
// ROI 边缘扩大的相对系尺寸。假设为0.5，那么左右多出宽为原图一半的边缘，图片尺寸扩大两倍
marginsize = 0.5
// 改成你的染色类型，和片子的文件名一致
def stain_types = ['HE', 'IHC']
// Process all HE images in the project.
def ImgName = GeneralTools.getNameWithoutExtension(server.getMetadata().getName())

for (stain_name in stain_types) {
    if (ImgName.contains(stain_name)) {
        WSI_ID = ImgName
    } else {
        continue
    }
    def Objects = getAnnotationObjects()
    def annotations = getAnnotationObjects().collect { new qupath.lib.objects.PathAnnotationObject(it.getROI(), it.getPathClass()) }
    // Write tiles
    def o = 0
    for (pathObject in Objects) {
        print(pathObject)
        if (pathObject.hasChildren()) {
            className = pathObject.getProperties().get('displayedName')
            print(className)
            def k = 0
            if (className == 'Kimura_sample') {
                //export tiles in the dir
                tiles = pathObject.getChildObjects()
                for (tile in tiles) {
                    roi = tile.getROI()

                    def sizey = roi.getBoundsHeight()
                    def sizex = roi.getBoundsWidth()
                    def sx = roi.getBoundsX()
                    // - (int)(sizex * marginsize)
                    def sy = roi.getBoundsY()

                    requestedTile = RegionRequest.createInstance(server.getPath(), downsample, roi)

                    name = "${ProjBaseDir}/${WSI_ID}_(d=$downsample, x=$sx, y=$sy, w=$sizex, h=$sizey, z=${k}).tif".toString()
                    print(roi)
                    writeImageRegion(server, requestedTile, name) //保存

                    k += 1
            }}
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
