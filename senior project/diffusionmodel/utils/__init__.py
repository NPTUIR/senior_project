# from .setpath import Datapath
# Datapath().path_process()
# print('init path')

def trainvideo(speed:int=1, imgpath:str='', result_name:str='output.mp4'):
    import cv2
    import glob
    assert imgpath!='', 'Path can not be empty!'
    # path = "./images/*.png"  # 所有png結尾的

    frame_list = sorted(glob.glob(imgpath))
    print("frame count: ", len(frame_list))

    fps = 20
    shape = cv2.imread(frame_list[0]).shape  # delete dimension 3
    size = (shape[1], shape[0])
    print("frame size: ", size)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(result_name, fourcc, fps, size)

    for idx, path in enumerate(frame_list):
        frame = cv2.imread(path)
        current_frame = idx + 1
        total_frame_count = len(frame_list)
        percentage = int(current_frame * 30 / (total_frame_count + 1))
        print("\rProcess: [{}{}] {:06d} / {:06d}".format("#" * percentage, "." * (30 - 1 - percentage), current_frame,
                                                         total_frame_count), end='')

        for i in range(speed):
            out.write(frame)

    out.release()
    print("Finish making video !!!")

if __name__=='__main__':
    trainvideo(speed=1, imgpath='../result/good/training/img/*.png')