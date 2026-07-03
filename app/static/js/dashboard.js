const modal = document.querySelector("#image-modal");
const modalImage = document.querySelector("#modal-image");
const modalCaption = document.querySelector("#modal-caption");
const closeButton = document.querySelector(".modal-close");

function closeModal() {
    if (!modal || !modalImage || !modalCaption) {
        return;
    }
    modal.classList.remove("open");
    modal.setAttribute("aria-hidden", "true");
    modalImage.src = "";
    modalImage.alt = "";
    modalCaption.textContent = "";
}

document.querySelectorAll(".zoomable-image").forEach((image) => {
    image.addEventListener("click", () => {
        if (!modal || !modalImage || !modalCaption) {
            return;
        }
        modalImage.src = image.src;
        modalImage.alt = image.alt;
        modalCaption.textContent = image.dataset.title || image.alt;
        modal.classList.add("open");
        modal.setAttribute("aria-hidden", "false");
    });
});

if (closeButton) {
    closeButton.addEventListener("click", closeModal);
}

if (modal) {
    modal.addEventListener("click", (event) => {
        if (event.target === modal) {
            closeModal();
        }
    });
}

document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
        closeModal();
    }
});
