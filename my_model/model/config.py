class DefaultConfig():
    # backbone
    pretrained = True
    freeze_stage_1 = True
    freeze_bn = True

    # fpn
    fpn_out_channels = 256
    use_p5 = True

    # head
    class_num = 80
    use_GN_head = True
    prior = 0.01
    add_centerness = True
    cnt_on_reg = False

    # training
    strides = [8, 16, 32, 64, 128]
    limit_range = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 999999]]

    # predict
    score_threshold = 0.3
    nms_iou_threshold = 0.2
    max_detection_boxes_num = 150
