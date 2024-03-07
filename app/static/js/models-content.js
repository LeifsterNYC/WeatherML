let currentSection = 0;
let oldCurrentSection = 0;
function loadSection(section) {
    oldCurrentSection = currentSection;
    currentSection = section;
    sessionStorage.setItem('currentSection', currentSection);
    const fileName = `static/html/model-sections/section${currentSection}.html`;
    fetch(fileName)
        .then(response => {
            if (!response.ok) {
                currentSection = oldCurrentSection;
                sessionStorage.setItem('currentSection', currentSection);
                throw new Error('No more sections.');
            }
            return response.text();
        })
        .then(html => {
            document.getElementById('body-container').innerHTML = html;
            Prism.highlightAll()
            initDivider()
        })
        .catch(error => {
            console.error('Error loading section:', error);

        });
}
