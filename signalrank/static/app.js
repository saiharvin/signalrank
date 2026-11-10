const state = {
  user: "U80234",
  sources: {}
}

const $ = id => document.getElementById(id)

const escapeHtml = value =>
  String(value).replace(
    /[&<>'"]/g,
    char =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        "'": "&#39;",
        '"': "&quot;"
      })[char]
  )

async function json(url, options = {}) {
  const response = await fetch(url, options)

  if (!response.ok) {
    const payload = await response.json()
    throw new Error(payload.detail || "Request failed")
  }

  return response.json()
}

async function loadUser(event) {
  if (event) {
    event.preventDefault()
  }

  state.user = $("user-id").value.trim()
  $("status").textContent = "Ranking…"

  try {
    const [profile, recommendations] = await Promise.all([
      json(`/profile/${encodeURIComponent(state.user)}`),
      json(`/recommend/${encodeURIComponent(state.user)}?k=6`)
    ])

    renderProfile(profile)
    renderRecommendations(recommendations.recommendations)

    $("status").textContent =
      `${recommendations.model} · ` +
      `${recommendations.recommendations.length} results`
  } catch (error) {
    $("status").textContent = error.message
  }
}

function renderProfile(profile) {
  $("profile").classList.remove("hidden")

  $("profile").innerHTML = `
    <p class="eyebrow">USER PROFILE</p>
    <h2>${escapeHtml(profile.user_id)}</h2>
    <p>
      ${profile.history_size} historical interactions ·
      ${profile.cold_start ? "Cold start" : "Personalized"}
    </p>

    <div class="profile-grid">
      ${profile.top_interests
        .map(
          item => `
            <span class="interest">
              <strong>${escapeHtml(item.category)}</strong>
              ${(item.affinity * 100).toFixed(0)}%
            </span>
          `
        )
        .join("")}
    </div>
  `
}

function renderRecommendations(items) {
  $("recommendations").innerHTML = items
    .map(
      item => `
        <article class="card">
          <span class="rank">
            RANK ${item.rank} · ${escapeHtml(item.category)}
          </span>

          <h3>${escapeHtml(item.title)}</h3>

          <div class="score">
            ${item.score.toFixed(3)}
          </div>

          <p>
            ${escapeHtml(
              item.abstract || "No abstract available in MIND."
            )}
          </p>

          ${item.reasons
            .map(
              reason => `
                <span class="reason">
                  ${escapeHtml(reason)}
                </span>
              `
            )
            .join("")}
        </article>
      `
    )
    .join("")
}

async function chat(event) {
  event.preventDefault()

  const question = $("question").value.trim()

  if (!question) {
    return
  }

  const container = $("conversation")

  container.innerHTML += `
    <div class="message">
      <strong>You</strong>
      <p>${escapeHtml(question)}</p>
    </div>
  `

  $("question").value = ""

  container.innerHTML += `
    <div class="message" id="thinking">
      Retrieving evidence…
    </div>
  `

  try {
    const answer = await json("/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        user_id: state.user,
        question,
        k: 5
      })
    })

    $("thinking").remove()

    state.sources = Object.fromEntries(
      answer.sources.map(source => [
        source.citation_id,
        source
      ])
    )

    const renderedClaims = answer.claims
      .map(
        claim => `
          <p>
            ${escapeHtml(claim.text)}

            ${claim.sources
              .map(
                source => `
                  <button
                    class="citation"
                    data-source="${escapeHtml(source)}"
                  >
                    [${escapeHtml(source)}]
                  </button>
                `
              )
              .join(" ")}
          </p>
        `
      )
      .join("")

    container.innerHTML += `
      <div class="message">
        <strong>SignalRank</strong>

        ${
          renderedClaims ||
          "<p>There was not enough evidence to answer.</p>"
        }

        <small>
          ${escapeHtml(answer.generation)} ·
          ${escapeHtml(answer.notice)}
        </small>
      </div>
    `

    $("provider").textContent = answer.generation
  } catch (error) {
    $("thinking").textContent = error.message
  }
}

document.addEventListener("click", event => {
  const citationId = event.target.dataset.source

  if (!citationId) {
    return
  }

  const source = state.sources[citationId]

  if (!source) {
    return
  }

  $("source-panel").classList.remove("hidden")

  $("source-content").innerHTML = `
    <h2>${escapeHtml(source.title)}</h2>

    <p class="eyebrow">
      ${escapeHtml(source.citation_id)} ·
      ${escapeHtml(source.category)}
    </p>

    <p>${escapeHtml(source.passage)}</p>
  `
})

$("close-source").onclick = () => {
  $("source-panel").classList.add("hidden")
}

$("user-form").onsubmit = loadUser
$("chat-form").onsubmit = chat

loadUser()