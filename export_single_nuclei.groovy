
import qupath.lib.images.servers.ImageServer
import qupath.lib.objects.PathObject

import javax.imageio.ImageIO
import java.awt.Color
import java.awt.image.BufferedImage

import qupath.lib.roi.RectangleROI
import qupath.lib.objects.PathAnnotationObject
import qupath.lib.objects.PathCellObject
import qupath.lib.objects.PathDetectionObject
import qupath.lib.objects.PathObject
import qupath.lib.objects.PathObjects
import qupath.lib.objects.PathTileObject
import qupath.lib.regions.RegionRequest
import qupath.lib.roi.RoiTools
import qupath.lib.roi.interfaces.ROI
import ij.plugin.ImageCalculator
import qupath.imagej.tools.IJTools
import java.awt.geom.AffineTransform

import static qupath.lib.gui.scripting.QPEx.*
import groovy.util.FileTreeBuilder
import static qupath.lib.gui.scripting.QPEx.*
import com.google.gson.GsonBuilder
import static qupath.lib.gui.scripting.QPEx.*


def he2ihcTransMatrices = [
        '01_17-7885_Ki67_': [1.0000, 0.0000, -4110.1113,
                             0.0000, 1.0000, -5672.8013], //ok
        '01_17-8107_Ki67_': [1.0000, 0.0000, -3619.2339,
                             -0.0000, 1.0000, -39.6530], //ok
        '01_14-3768_Ki67_': [1.0000, 0.0000, -4099.9741,
                             0.0000, 1.0000, -51.2197],//ok
        '01_14-7015_Ki67_': [1.0000, 0.0000, -3919.6276,
                             0.0000, 1.0000, 37.3973], //ok
        '01_15-1052_Ki67_': [1.0000, -0.0000, 860.3158,
                             0.0000, 1.0000, -1432.6411], //ok
        '01_15-2502_Ki67_': [1.0000, 0.0000, -1052.4520,
                             0.0000, 1.0000, -15.0699], //ok
        '01_17-5256_Ki67_': [1.0000, 0.0000, -5321.4238,
                             0.0000, 1.0000, -3674.8283], //ok
        '01_17-6747_Ki67_': [1.0000, 0.0000, 279.7157,
                             0.0000, 1.0000, -1757.3694], //ok
        '01_17-7930_Ki67_': [1.0000, 0.0000, -606.0000,
                             0.0000, 1.0000, 1468.9137] //ok
]

def roiSize = 64

for (WSI_ID in he2ihcTransMatrices.keySet()) {
    def ihcAnnotImgName = WSI_ID + 'IHC.ndpi'
    def heTargetImgName = WSI_ID + 'HE.ndpi'

    def matrix = he2ihcTransMatrices[WSI_ID]


    def imageData = project.getImageList().find { it.getImageName() == ihcAnnotImgName }.readImageData()
    def server = imageData.getServer()
    def ImgName = getProjectEntry().getImageName()
    if (ImgName != ihcAnnotImgName) {continue}
    def cells = getCellObjects()

    def he_entry = project.getImageList().find { it.getImageName() == heTargetImgName }
    def heData = he_entry.readImageData()
    def heserver = heData.getServer()

    // TODO: 限定在打标签的采样区域。和之前的tumor标记同

    k=0
    // Detect the cells in desired regions in advance
    // 预先完成ROI中的细胞检测。只会对检测对象进行操作
    new File("/Users/cunyuan/DATA/${WSI_ID}").mkdirs()
    for (cell in cells) {
        roi = cell.getROI()
        double cx = roi.getCentroidX()
        double cy = roi.getCentroidY()
        print("x: ${cx}, y: ${cy}")

        // 格式化输出文件名。请按需修改
        // 原始ROI（无重叠）
        name = "/Users/cunyuan/DATA/${WSI_ID}/${k}i.tif".toString()
        def roi = new RectangleROI(cx, cy, roiSize, roiSize)
        print(roi)

        requestedTile = RegionRequest.createInstance(server.getPath(), 1, roi)

        writeImageRegion(server, requestedTile, name) //保存

        def transform = new AffineTransform(
            matrix[0] as double, matrix[3] as double, matrix[1] as double,
            matrix[4] as double, matrix[2] as double, matrix[5] as double
        )
        transform = transform.createInverse()
        roi1 = transformROI(roi, transform)
        requestedTile = RegionRequest.createInstance(server.getPath(), 1, roi1)
        name = "/Users/cunyuan/DATA/${WSI_ID}/${k}h.tif".toString()
        print(name)
        writeImageRegion(heserver, requestedTile, name) //保存
        k = k+1
    }
}

/**
 * Transform ROI (via conversion to Java AWT shape)
 *
 * @param roi
 * @param transform
 * @return
 */
ROI transformROI(ROI roi, AffineTransform transform) {
    def shape = RoiTools.getShape(roi)
    // Should be able to use roi.getShape() - but there's currently a bug in it for rectangles/ellipses!
    shape2 = transform.createTransformedShape(shape)
    return RoiTools.getShapeROI(shape2, roi.getImagePlane(), 0.5)
}