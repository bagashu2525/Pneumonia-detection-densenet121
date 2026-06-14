# ============================================
# PHASE 3 : TRAINING
# phase3.py
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
    scheduler,
    device,
    epochs=10,
    patience=5,
    model_path="/content/best_pneumonia_model.pth"
):

    # ========================================
    # HISTORY STORAGE
    # ========================================

    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []

    # ========================================
    # EARLY STOPPING
    # ========================================

    best_loss = float("inf")

    patience_counter = 0

    # ========================================
    # TRAINING TIMER
    # ========================================

    start_time = time.time()

    print("\n===================================")
    print("TRAINING STARTED")
    print("===================================")

    # ========================================
    # EPOCH LOOP
    # ========================================

    for epoch in range(epochs):

        # ====================================
        # TRAINING
        # ====================================

        model.train()

        running_train_loss = 0

        train_correct = 0
        train_total = 0

        for images, labels in train_loader:

            images = images.to(device)

            labels = (
                labels
                .float()
                .unsqueeze(1)
                .to(device)
            )

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_train_loss += loss.item()

            preds = (
                torch.sigmoid(outputs) > 0.5
            ).float()

            train_correct += (
                preds == labels
            ).sum().item()

            train_total += labels.size(0)

        # ====================================
        # VALIDATION
        # ====================================

        model.eval()

        running_val_loss = 0

        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)

                labels = (
                    labels
                    .float()
                    .unsqueeze(1)
                    .to(device)
                )

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                running_val_loss += loss.item()

                preds = (
                    torch.sigmoid(outputs) > 0.5
                ).float()

                val_correct += (
                    preds == labels
                ).sum().item()

                val_total += labels.size(0)

        # ====================================
        # EPOCH METRICS
        # ====================================

        train_loss = (
            running_train_loss
            / len(train_loader)
        )

        val_loss = (
            running_val_loss
            / len(val_loader)
        )

        train_acc = (
            train_correct
            / train_total
        )

        val_acc = (
            val_correct
            / val_total
        )

        # ====================================
        # STORE HISTORY
        # ====================================

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # ====================================
        # LEARNING RATE SCHEDULER
        # ====================================

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        # ====================================
        # PRINT RESULTS
        # ====================================

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.8f}"
        )

        # ====================================
        # SAVE BEST MODEL
        # ====================================

        if val_loss < best_loss:

            best_loss = val_loss

            torch.save(
                model.state_dict(),
                model_path
            )

            patience_counter = 0

            print(
                "[OK] Best Model Saved"
            )

        else:

            patience_counter += 1

        # ====================================
        # EARLY STOPPING
        # ====================================

        if patience_counter >= patience:

            print(
                f"\n[INFO] Early Stopping Triggered "
                f"after {epoch+1} epochs"
            )

            break

    # ========================================
    # TRAINING TIME
    # ========================================

    end_time = time.time()

    training_time = (
        end_time - start_time
    )

    print("\n===================================")
    print("TRAINING COMPLETED")
    print("===================================")

    print(
        f"Total Training Time: "
        f"{training_time:.2f} seconds"
    )

    print(
        f"Best Validation Loss: "
        f"{best_loss:.4f}"
    )

    # ========================================
    # RETURN EVERYTHING
    # ========================================

    history = {

        "train_losses":
            train_losses,

        "val_losses":
            val_losses,

        "train_accuracies":
            train_accuracies,

        "val_accuracies":
            val_accuracies,

        "training_time":
            training_time,

        "best_val_loss":
            best_loss
    }

    return history