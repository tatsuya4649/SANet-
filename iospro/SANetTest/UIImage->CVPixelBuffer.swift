//
//  UIImage->CVPixelBuffer.swift
//  WCTforSwift
//
//  Created by 下川達也 on 2020/07/02.
//  Copyright © 2020 下川達也. All rights reserved.
//

import Foundation
import UIKit
import CoreML

extension UIImage {
    func resizeStyle(to newSize: CGSize) -> UIImage {
        UIGraphicsBeginImageContextWithOptions(CGSize(width: newSize.width, height: newSize.height), true, 1.0)
        if self.size.width > self.size.height{
            self.draw(in: CGRect(x: newSize.width/2 - (newSize.width * self.size.width/self.size.height)/2, y: 0, width: (newSize.width * self.size.width/self.size.height), height: newSize.height))
        }else{
            self.draw(in: CGRect(x: 0, y: newSize.height/2 - (newSize.height * self.size.height/self.size.width)/2, width: newSize.width, height: newSize.height * self.size.height/self.size.width))
        }
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()
        return resizedImage
    }
    
    func resizeContent(to newSize: CGSize) -> UIImage {
        UIGraphicsBeginImageContextWithOptions(CGSize(width: newSize.width, height: newSize.height), true, 1.0)
        if self.size.width > self.size.height{
            self.draw(in: CGRect(x: 0, y: newSize.height/2 - (newSize.height * (self.size.height/self.size.width))/2, width: newSize.width, height: newSize.height * (self.size.height/self.size.width)))
        }else{
            self.draw(in: CGRect(x: newSize.width/2 - (newSize.width * (self.size.width/self.size.height))/2, y: 0, width: newSize.width * (self.size.width/self.size.height), height: newSize.height))
        }
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()
        return resizedImage
    }
    
    func resizeCrop(resize:CGSize) -> UIImage{
        UIGraphicsBeginImageContextWithOptions(CGSize(width: resize.width, height: resize.height), true, 1.0)
        if resize.width > resize.height{
            let height = self.size.height * resize.width/self.size.width
            self.draw(in: CGRect(x: resize.height/2 - height/2, y: 0,width: resize.width, height: height))
        }else{
            print()
            let width = self.size.width * resize.height/self.size.height
            self.draw(in: CGRect(x: resize.width/2 - width/2, y: 0,width: width, height: resize.height))
        }
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()
        return resizedImage
    }

    func pixelBuffer() -> CVPixelBuffer? {
        let width = self.size.width
        let height = self.size.height
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                     kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                         Int(width),
                                         Int(height),
                                         kCVPixelFormatType_32ARGB,
                                         attrs,
                                         &pixelBuffer)

        guard let resultPixelBuffer = pixelBuffer, status == kCVReturnSuccess else {
            return nil
        }

        CVPixelBufferLockBaseAddress(resultPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(resultPixelBuffer)

        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(data: pixelData,
                                      width: Int(width),
                                      height: Int(height),
                                      bitsPerComponent: 8,
                                      bytesPerRow: CVPixelBufferGetBytesPerRow(resultPixelBuffer),
                                      space: rgbColorSpace,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue) else {
                                        return nil
        }

        context.translateBy(x: 0, y: height)
        context.scaleBy(x: 1.0, y: -1.0)

        UIGraphicsPushContext(context)
        self.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
        UIGraphicsPopContext()
        CVPixelBufferUnlockBaseAddress(resultPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        
        return resultPixelBuffer
    }
    public func pixelData() -> [UInt8]? {
        let dataSize = size.width * size.height * 4
        var pixelData = [UInt8](repeating: 0, count: Int(dataSize))
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: &pixelData, width: Int(size.width), height: Int(size.height), bitsPerComponent: 8, bytesPerRow: 4 * Int(size.width), space: colorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        
        guard let cgImage = self.cgImage else { return nil }
        context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: size.width, height: size.height))
        return pixelData
    }
}

#if canImport(UIKit)

import UIKit

extension UIImage {
  /**
    Converts the image into an array of RGBA bytes.
  */
  @nonobjc public func toByteArrayRGBA() -> [UInt8]? {
    return cgImage?.toByteArrayRGBA()
  }

  /**
    Creates a new UIImage from an array of RGBA bytes.
  */
  @nonobjc public class func fromByteArrayRGBA(_ bytes: [UInt8],
                                               width: Int,
                                               height: Int,
                                               scale: CGFloat = 0,
                                               orientation: UIImage.Orientation = .up) -> UIImage? {
    if let cgImage = CGImage.fromByteArrayRGBA(bytes, width: width, height: height) {
      return UIImage(cgImage: cgImage, scale: scale, orientation: orientation)
    } else {
      return nil
    }
  }

  /**
    Creates a new UIImage from an array of grayscale bytes.
  */
  @nonobjc public class func fromByteArrayGray(_ bytes: [UInt8],
                                               width: Int,
                                               height: Int,
                                               scale: CGFloat = 0,
                                               orientation: UIImage.Orientation = .up) -> UIImage? {
    if let cgImage = CGImage.fromByteArrayGray(bytes, width: width, height: height) {
      return UIImage(cgImage: cgImage, scale: scale, orientation: orientation)
    } else {
      return nil
    }
  }
}

#endif

import CoreGraphics

extension CGImage {
  /**
    Converts the image into an array of RGBA bytes.
  */
  @nonobjc public func toByteArrayRGBA() -> [UInt8] {
    var bytes = [UInt8](repeating: 0, count: width * height * 4)
    bytes.withUnsafeMutableBytes { ptr in
      if let colorSpace = colorSpace,
         let context = CGContext(
                    data: ptr.baseAddress,
                    width: width,
                    height: height,
                    bitsPerComponent: bitsPerComponent,
                    bytesPerRow: bytesPerRow,
                    space: colorSpace,
                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) {
        let rect = CGRect(x: 0, y: 0, width: width, height: height)
        context.draw(self, in: rect)
      }
    }
    return bytes
  }

  /**
    Creates a new CGImage from an array of RGBA bytes.
  */
  @nonobjc public class func fromByteArrayRGBA(_ bytes: [UInt8],
                                               width: Int,
                                               height: Int) -> CGImage? {
    return fromByteArray(bytes, width: width, height: height,
                         bytesPerRow: width * 4,
                         colorSpace: CGColorSpaceCreateDeviceRGB(),
                         alphaInfo: .premultipliedLast)
  }

  /**
    Creates a new CGImage from an array of grayscale bytes.
  */
  @nonobjc public class func fromByteArrayGray(_ bytes: [UInt8],
                                               width: Int,
                                               height: Int) -> CGImage? {
    return fromByteArray(bytes, width: width, height: height,
                         bytesPerRow: width,
                         colorSpace: CGColorSpaceCreateDeviceGray(),
                         alphaInfo: .none)
  }

  @nonobjc class func fromByteArray(_ bytes: [UInt8],
                                    width: Int,
                                    height: Int,
                                    bytesPerRow: Int,
                                    colorSpace: CGColorSpace,
                                    alphaInfo: CGImageAlphaInfo) -> CGImage? {
    return bytes.withUnsafeBytes { ptr in
        let context = CGContext(data: UnsafeMutableRawPointer(mutating: ptr.baseAddress!),
                              width: width,
                              height: height,
                              bitsPerComponent: 8,
                              bytesPerRow: bytesPerRow,
                              space: colorSpace,
                              bitmapInfo: alphaInfo.rawValue)
      return context?.makeImage()
    }
  }
}

extension UIImage {
    //pixelBUfferに変換(RGB値)
    func getPixelRgb() -> [Float32]
    {
        guard let cgImage = self.cgImage else {
            return []
        }
        let bytesPerRow = cgImage.bytesPerRow
        let width = cgImage.width
        let height = cgImage.height
        let bytesPerPixel = 4
        let pixelData = cgImage.dataProvider!.data! as Data
        //print(pixelData.count)
        var r_buf : [Float32] = []
        var g_buf : [Float32] = []
        var b_buf : [Float32] = []

        var h_buf : [[Float32]] = []
        var w_buf : [Float32] = []
        for j in 0..<height {
            for i in 0..<width {
                let pixelInfo = bytesPerRow * j + i * bytesPerPixel
                let r = Float32(pixelData[pixelInfo+2])
                let g = Float32(pixelData[pixelInfo+1])
                let b = Float32(pixelData[pixelInfo])
                //r_buf.append(Float32(r/255.0))
                //g_buf.append(Float32(g/255.0))
                //b_buf.append(Float32(b/255.0))
                //rbg,//rgb,//brg,//bgr,//gbr,//grb
                w_buf.append(Float32(r))
                w_buf.append(Float32(g))
                w_buf.append(Float32(b))
            }
            //h_buf.append(w_buf)
        }
        print(pixelData.count)
        return w_buf//((b_buf + g_buf) + r_buf)
    }
}

