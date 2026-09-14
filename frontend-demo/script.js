const sendBtn = document.getElementById("send-btn");
const questionInput = document.getElementById("question");
const responseBox = document.getElementById("response");
const statusBox = document.getElementById("status");
const suggestionsBox = document.getElementById("suggestions");

const API_URL = "http://localhost:8000/api/v1/chat";


sendBtn.addEventListener("click", function (event) {
  event.preventDefault();
  sendQuestion();
});


async function sendQuestion() {
  const question = questionInput.value.trim();

  if (!question) {
    responseBox.innerHTML = "Merci de saisir une question.";
    statusBox.textContent = "";
    suggestionsBox.innerHTML = "";
    return;
  }

  await callApi({
    question: question,
    top_k: 5
  });

  questionInput.value = "";
}


async function sendChoice(choice) {
  await callApi({
    choice: choice,
    top_k: 5
  });
}


async function loadMainMenu() {
  await callApi({
    question: "",
    choice: "",
    top_k: 5
  });
}


async function callApi(payload) {
  statusBox.textContent = "Chargement...";
  responseBox.innerHTML = "Chargement de la réponse...";
  suggestionsBox.innerHTML = "";

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });

    console.log("Status API:", response.status);

    if (!response.ok) {
      throw new Error(`Erreur API : ${response.status}`);
    }

    const data = await response.json();

    console.log("Réponse API :", data);

    const markdownText =
      data.answer || "Aucune réponse reçue.";

    responseBox.innerHTML = marked.parse(markdownText);

    displaySuggestions(data.suggestions || []);

    statusBox.textContent = "Réponse reçue avec succès.";

  } catch (error) {
    console.error("Erreur fetch:", error);

    responseBox.innerHTML =
      "Erreur lors de l'appel API.";

    statusBox.textContent = "Échec.";

    suggestionsBox.innerHTML = "";
  }
}


function displaySuggestions(suggestions) {
  suggestionsBox.innerHTML = "";

  suggestions.forEach((suggestion) => {
    const button = document.createElement("button");

    button.type = "button";
    button.className = "suggestion-btn";

    button.textContent = suggestion.label;

    button.addEventListener("click", () => {
      sendChoice(suggestion.value);
    });

    suggestionsBox.appendChild(button);
  });
}


loadMainMenu();