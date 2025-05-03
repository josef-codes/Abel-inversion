% Ref orez
i2 = imread("refbmp.bmp");
filename = "Conversion_sample.bmp";
imwrite(i2(116:238,289:519),filename,'bmp')
% load phase shift
i1 = imread("1,5usGD_100nsGW_230mcp___X8.tif");
%
dbIm16 = double(i1)+1;
db16min = min(dbIm16(:)); db16max = max(dbIm16(:));
TgBit = 8; % or any other lower bit scale
% example with 16bit to 8bit
Norm_wOffSet = dbIm16/db16max; % maintaining putative offset from 0 in the data
Im8_wOffSet = uint8(Norm_wOffSet*2^TgBit-1); % back to 0:2^8-1
Norm_woOffSet = (dbIm16-db16min)/(db16max-db16min); % Scales linearly to full range
Im8_woOffSet = uint8(Norm_woOffSet*2^TgBit-1); % back to 0:2^8-1
figure; imshow(Im8_woOffSet)
%
outpict = im2uint8(i1); % recast with appropriate scaling
figure; imshow(outpict(116:238,289:519))
%
filename = "Conversion_X8.bmp";
imwrite(outpict(116:238,289:519),filename,'bmp')
