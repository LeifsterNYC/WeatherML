document.addEventListener("DOMContentLoaded", function () {
    const navBarContainer = document.getElementById("navbar");
    if (navBarContainer) {
        fetch('static/html/nav.html')
            .then(response => response.text())
            .then(data => {
                navBarContainer.innerHTML = data;
                const currentPath = window.location.pathname;
                const navLinks = document.querySelectorAll('.top-nav a');
                navLinks.forEach(link => {
                    if (link.getAttribute('href') === currentPath) {
                        link.classList.add('active');
                    } else {
                        link.classList.remove('active');
                    }
                });
            });
    }
});
