
function Link(el)
  if el.target:match("%.pdf$") then
    el.classes = {"btn", "btn-primary", "btn-sm"}
    el.content = {pandoc.Str("Paper")}
  elseif el.target:match("github.com") then
    el.classes = {"btn", "btn-success", "btn-sm"}
    el.content = {pandoc.Str("Code")}
  elseif el.target:match("poster") then
    el.classes = {"btn", "btn-info", "btn-sm"}
    el.content = {pandoc.Str("Poster")}
  end
  return el
end
