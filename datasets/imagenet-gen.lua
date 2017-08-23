--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Script to compute list of ImageNet filenames and classes
--
--  This generates a file gen/imagenet.t7 which contains the list of all
--  ImageNet training and validation images and their classes. This script also
--  works for other datasets arragned with the same layout.
--

local sys = require 'sys'
local ffi = require 'ffi'

local M = {}

local function findClasses(dir)
   local dirs = paths.dir(dir)
   table.sort(dirs)

   local classList = {}
   local classToIdx = {}
   for _ ,class in ipairs(dirs) do
      if not classToIdx[class] and class ~= '.' and class ~= '..' and class ~= '.DS_Store' then
         table.insert(classList, class)
         classToIdx[class] = #classList
      end
   end

   -- assert(#classList == 1000, 'expected 1000 ImageNet classes')
   return classList, classToIdx
end

local function findImages(dir, classToIdx)
   local imagePath = torch.CharTensor()
   local imageClass = torch.LongTensor()

   ----------------------------------------------------------------------
   -- Options for the GNU and BSD find command
   local extensionList = {'jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
   local findOptions = ' -iname "*.' .. extensionList[1] .. '"'
   for i=2,#extensionList do
      findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
   end

   -- Find all the images using the find command
   local f = io.popen('find -L ' .. dir .. findOptions)

   local maxLength = -1
   local imagePaths = {}
   local imageClasses = {}

   -- Generate a list of all the images and their class
   while true do
      local line = f:read('*line')
      if not line then break end

      local className = paths.basename(paths.dirname(line))
      local filename = paths.basename(line)
      local path = className .. '/' .. filename

      local classId = classToIdx[className]
      assert(classId, 'class not found: ' .. className)

      table.insert(imagePaths, path)
      table.insert(imageClasses, classId)

      maxLength = math.max(maxLength, #path + 1)
   end

   f:close()

   -- Convert the generated list to a tensor for faster loading
   local nImages = #imagePaths
   local imagePath = torch.CharTensor(nImages, maxLength):zero()
   for i, path in ipairs(imagePaths) do
      ffi.copy(imagePath[i]:data(), path)
   end

   local imageClass = torch.LongTensor(imageClasses)
   return imagePath, imageClass
end

function M.exec(opt, cacheFile)
   -- find the image path names
   local imagePath = torch.CharTensor()  -- path to each image in dataset
   local imageClass = torch.LongTensor() -- class index of each image (class index in self.classes)

   local trainDir = paths.concat(opt.data, 'train')
   local valDir = paths.concat(opt.data, 'val')
   assert(paths.dirp(trainDir), 'train directory not found: ' .. trainDir)
   assert(paths.dirp(valDir), 'val directory not found: ' .. valDir)

   print("=> Generating list of images")
   local classList, classToIdx = findClasses(trainDir)

   print(" | finding all validation images")
   local valImagePath, valImageClass = findImages(valDir, classToIdx)

   print(" | finding all training images")
   local trainImagePath, trainImageClass = findImages(trainDir, classToIdx)

   local info = {
      basedir = opt.data,
      classList = classList,
      train = {
         imagePath = trainImagePath,
         imageClass = trainImageClass,
      },
      val = {
         imagePath = valImagePath,
         imageClass = valImageClass,
      },
   }

   print(" | saving list of images to " .. cacheFile)
   torch.save(cacheFile, info)
   return info
end

return M
