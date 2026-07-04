// Elementos del visor grande de imagenes.
const modal = document.querySelector("#image-modal");
const modalImage = document.querySelector("#modal-image");
const modalCaption = document.querySelector("#modal-caption");
const closeButton = document.querySelector(".modal-close");

function closeModal() {
    // Si la pagina no tiene modal, no hace nada y evita errores.
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
        // Al hacer click en una grafica, se copia su imagen al visor grande.
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
    // Boton X para cerrar el visor.
    closeButton.addEventListener("click", closeModal);
}

if (modal) {
    modal.addEventListener("click", (event) => {
        // Click fuera de la imagen tambien cierra el visor.
        if (event.target === modal) {
            closeModal();
        }
    });
}

document.addEventListener("keydown", (event) => {
    // Escape cierra el visor sin usar el mouse.
    if (event.key === "Escape") {
        closeModal();
    }
});

// Overlay de carga: se muestra cuando el usuario envia un formulario o hace click en un
// enlace que dispara un recalculo en el servidor (pandas/sklearn corriendo en ese momento).
const loadingOverlay = document.querySelector("#loading-overlay");

function showLoadingOverlay() {
    if (loadingOverlay) {
        loadingOverlay.classList.add("visible");
    }
}

document.querySelectorAll("form.js-loading-trigger").forEach((form) => {
    form.addEventListener("submit", showLoadingOverlay);
});

document.querySelectorAll("a.js-loading-trigger").forEach((link) => {
    link.addEventListener("click", showLoadingOverlay);
});
