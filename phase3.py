# ============================================
# PHASE 3 : TRAINING FUNCTION
# ============================================

import torch

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=10
):

    best_loss = float("inf")

    for epoch in range(epochs):

        # ====================================
        # TRAINING
        # ====================================

        model.train()

        train_loss = 0

        for images, labels in train_loader:

            images = images.to(device)

            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        # ====================================
        # VALIDATION
        # ====================================

        model.eval()

        val_loss = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)

                labels = labels.float().unsqueeze(1).to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                val_loss += loss.item()

        train_loss /= len(train_loader)

        val_loss /= len(val_loader)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f}"
        )

        # ====================================
        # SAVE BEST MODEL
        # ====================================

        if val_loss < best_loss:

            best_loss = val_loss

            torch.save(
                model.state_dict(),
                "best_pneumonia_model.pth"
            )

            print("✅ Best Model Saved")

    print("\n✅ Training Finished")