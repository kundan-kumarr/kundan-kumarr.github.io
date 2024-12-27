---
layout: archive
title: "Projects"
permalink: /projects/
author_profile: true
---

<style>
.projects-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 1.5rem;
  margin-top: 2rem;
}

.project-card {
  border: 1px solid #e0e0e0;
  border-radius: 4px;
  overflow: hidden;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.project-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.project-image {
  width: 100%;
  height: 200px;
  object-fit: cover;
}

.project-content {
  padding: 1rem;
}

.project-title {
  margin: 0 0 0.5rem 0;
  font-size: 1.2rem;
  color: #333;
}

.project-title a {
  text-decoration: none;
  color: inherit;
}

.project-title a:hover {
  color: #0366d6;
}

.project-excerpt {
  margin: 0;
  font-size: 0.9rem;
  color: #666;
  line-height: 1.5;
}
</style>

<div class="projects-grid">
  {% for post in site.projects %}
    <div class="project-card">
      {% if post.header.teaser %}
        <img src="{{ post.header.teaser | relative_url }}" alt="{{ post.title }}" class="project-image">
      {% endif %}
      <div class="project-content">
        <h2 class="project-title">
          <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
        </h2>
        {% if post.excerpt %}
          <p class="project-excerpt">{{ post.excerpt | strip_html | truncate: 160 }}</p>
        {% endif %}
      </div>
    </div>
  {% endfor %}
</div>