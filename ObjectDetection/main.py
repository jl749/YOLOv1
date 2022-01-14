from ObjectDetection.utils.functions import cellboxes_to_boxes, non_max_suppression, plot_image, save_checkpoint
from train import *


def main():
    model: nn.Module = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset: torch.utils.data.Dataset = VOCDataset(
        DATA_DIR.joinpath("100examples.csv"),
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    test_dataset = VOCDataset(
        DATA_DIR.joinpath("test.csv"), transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,  # instead of padding batch drop it
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    for epoch in range(EPOCHS):
        # for x, y in train_loader:
        #     x = x.to(DEVICE)
        #     for idx in range(8):
        #         bboxes = cellboxes_to_boxes(model(x))
        #         bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
        #         plot_image(x[idx].permute(1, 2, 0).to("cpu"), bboxes)
        #
        #     import sys
        #     sys.exit()

        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_prec}")

        # if mean_avg_prec > 0.9:
        #     checkpoint = {
        #         "state_dict": model.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #     }
        #     save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
        #     import time
        #     time.sleep(10)

        train(train_loader, model, optimizer, loss_fn)


if __name__ == '__main__':
    main()
