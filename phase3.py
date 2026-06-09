# ============================================
# PHASE 3 : TRAINING FUNCTION
# ============================================

import torch
import time

# ============================================
# TRAIN FUNCTION
# ============================================

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=20
):

    best_loss = float("inf")

    # ====================================
    # HISTORY
    # ====================================

    history = {

        "train_loss": [],
        "val_loss": [],

        "train_acc": [],
        "val_acc": [],

        "epoch_times": []
    }

    # ====================================
    # EPOCH LOOP
    # ====================================

    for epoch in range(epochs):

        epoch_start = time.time()

        # ====================================
        # TRAINING
        # ====================================

        model.train()

        running_train_loss = 0.0

        train_correct = 0
        train_total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_train_loss += loss.item()

            preds = torch.argmax(
                outputs,
                dim=1
            )

            train_correct += (
                preds == labels
            ).sum().item()

            train_total += labels.size(0)

        train_loss = (
            running_train_loss /
            len(train_loader)
        )

        train_acc = (
            train_correct /
            train_total
        )

        # ====================================
        # VALIDATION
        # ====================================

        model.eval()

        running_val_loss = 0.0

        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.long().to(device)

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                running_val_loss += loss.item()

                preds = torch.argmax(
                    outputs,
                    dim=1
                )

                val_correct += (
                    preds == labels
                ).sum().item()

                val_total += labels.size(0)

        val_loss = (
            running_val_loss /
            len(val_loader)
        )

        val_acc = (
            val_correct /
            val_total
        )

        # ====================================
        # EPOCH TIME
        # ====================================

        epoch_time = (
            time.time() -
            epoch_start
        )

        # ====================================
        # SAVE HISTORY
        # ====================================

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        history["train_acc"].append(
            train_acc * 100
        )

        history["val_acc"].append(
            val_acc * 100
        )

        history["epoch_times"].append(
            epoch_time
        )

        # ====================================
        # PRINT METRICS
        # ====================================

        print(
            f"\nEpoch [{epoch+1}/{epochs}]"
        )

        print(
            f"Train Loss : {train_loss:.4f}"
        )

        print(
            f"Val Loss   : {val_loss:.4f}"
        )

        print(
            f"Train Acc  : {train_acc*100:.2f}%"
        )

        print(
            f"Val Acc    : {val_acc*100:.2f}%"
        )

        print(
            f"Time       : {epoch_time:.2f} sec"
        )

        # ====================================
        # SAVE BEST MODEL
        # ====================================

        if val_loss < best_loss:

            best_loss = val_loss

            torch.save(
                model.state_dict(),
                "best_multiclass_model.pth"
            )

            print(
                "[OK] Best Model Saved"
            )

    # ====================================
    # TRAINING COMPLETE
    # ====================================

    print("\n" + "=" * 60)
    print("TRAINING FINISHED")
    print("=" * 60)

    print(
        f"Best Validation Loss : "
        f"{best_loss:.4f}"
    )

    # ====================================
    # RETURN HISTORY
    # ====================================

    return history