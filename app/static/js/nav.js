document.addEventListener("DOMContentLoaded", function () {
    const navBarContainer = document.getElementById("navbar");
    if (navBarContainer) {
        fetch('static/html/nav.html')
            .then(response => response.text())
            .then(data => {
                navBarContainer.innerHTML = data;
                const navLinks = document.querySelectorAll(".top-nav a");
                navLinks.forEach(link => {
                    link.addEventListener("click", function (event) {
                        navLinks.forEach(navLink => {
                            navLink.classList.remove("active");
                        });
                        link.classList.add("active");
                    });
                });
            });
    }
});
