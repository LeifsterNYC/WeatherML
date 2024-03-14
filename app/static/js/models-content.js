let currentSection = 0;
let oldCurrentSection = 0;
let sections = ['about', 'lr', 'lrts', 'rf', 'deep'];
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
            Prism.highlightAll();
            newSection(sections[currentSection]);
            loadGraph(1);
            window.scrollTo(0, 0);
        })
        .catch(error => {
            console.error('Error loading section:', error);

        });
}

let currentGraph = 1;
let oldCurrentGraph = 1;
let currentSectionName = 'about';

const sectionGraphs = {
    about: ['item1', 'item2', 'item3'],
    lr: ['lr1.html', 'lr2.html', 'lr3.html', 'lr4.html', 'lr5.html'],
    lrts: ['lrts1.html', 'lrts2.html', 'lrts3.html', 'lrts4.html', 'lrts5.html', 'lrts6.html'],
    rf: ['rf1.html', 'rf2.html', 'rf3.html', 'rf4.html', 'rf5.html', 'rf6.html', 'rf7.html', 'rf8.html', 'rf9.html', 'rf10.html'],
    deep: ['deep1.html']
};

function loadGraph(g) {
    currentGraph = g;
    file = sectionGraphs[currentSectionName][currentGraph - 1];
    const fileName = `static/graphs/${file}`;
    fetch(fileName)
        .then(response => {
            if (!response.ok) {
                currentGraph = oldCurrentGraph;
                throw new Error('No more Graphs.');
            }
            document.getElementById(`graph-${currentSectionName}-left`).style.visibility = currentGraph > 1 ? 'visible' : 'hidden';
            document.getElementById(`graph-${currentSectionName}-right`).style.visibility = currentGraph < sectionGraphs[currentSectionName].length ? 'visible' : 'hidden';
            return response.text();
        })
        .then(html => {
            document.getElementById('graphFrame').setAttribute("src", fileName);
            oldCurrentGraph = g;
        })
        .catch(error => {
            console.error('Error loading Graph:', error);

        });
}

function loadPreviousGraph() {
    loadGraph(currentGraph - 1)
}

function loadNextGraph() {
    loadGraph(currentGraph + 1)
}
function newSection(section) {
    currentGraph = 1;
    oldCurrentGraph = 1;
    currentSectionName = section;
}

window.addEventListener()