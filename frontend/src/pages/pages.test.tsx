import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { api } from '../api/client'
import { PostBadge } from '../components/ui'
import Agent from './Agent'
import Annotate from './Annotate'
import Topics from './Topics'

vi.mock('../api/client', () => {
  const fn = () => vi.fn()
  return {
    keyStore: { get: () => '', set: () => {} },
    errorMessage: (err: unknown, fallback: string) => (err instanceof Error ? err.message : fallback),
    api: {
      topics: fn(), createTopic: fn(), deleteTopic: fn(), posts: fn(), trends: fn(), drift: fn(), alerts: fn(),
      ackAlert: fn(), insights: fn(), runInsights: fn(), ingestTopic: fn(), startAgent: fn(), agentRun: fn(),
      annotationQueue: fn(), refreshQueue: fn(), exportQueue: fn(),
    },
  }
})

const m = api as unknown as Record<string, ReturnType<typeof vi.fn>>

const TOPIC = { id: 1, name: 'Fuel prices', query: 'petrol diesel', sources: ['google_news'], active: true, posts: 12, last_ingested: null }
const WINDOW = { n: 30, counts: { positive: 5, neutral: 10, negative: 15, uncertain: 0 }, mean_confidence: 0.9, uncertain_share: 0.05, oov_rate: 0.35 }
const DRIFT = { window_hours: 24, window: WINDOW, baseline: { ...WINDOW, n: 100, oov_rate: 0.334, days_covered: 6 }, psi: 0.31 }
const ALERT = { id: 7, kind: 'negative_spike', message: 'Negative share rose from 20% to 40%.', baseline: 0.2, observed: 0.4, p_value: 0.002, acknowledged: false, created_at: '2026-10-30T08:00:00' }

beforeAll(() => {
  // recharts measures its container with ResizeObserver, which jsdom does not have
  globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} }
})

beforeEach(() => {
  Object.values(m).forEach((f) => f.mockReset())
  m.topics.mockResolvedValue([TOPIC])
  m.alerts.mockResolvedValue([ALERT])
  m.drift.mockResolvedValue(DRIFT)
  m.trends.mockResolvedValue([])
  m.posts.mockResolvedValue({ total: 0, page: 1, limit: 15, total_pages: 0, posts: [] })
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

const openTopic = async () => {
  render(<Topics />)
  fireEvent.click(await screen.findByRole('button', { name: /Fuel prices/ }))
  await screen.findByRole('heading', { name: 'Fuel prices' })
}

describe('Topics', () => {
  it('adds a topic with trimmed values and selects it', async () => {
    m.createTopic.mockResolvedValue({ id: 2 })
    render(<Topics />)
    await screen.findByText('12 posts · not collected yet')
    fireEvent.change(screen.getByPlaceholderText('Name (e.g. Fuel prices)'), { target: { value: '  Rail  ' } })
    fireEvent.change(screen.getByPlaceholderText('Search query'), { target: { value: ' railway delay ' } })
    fireEvent.click(screen.getByRole('button', { name: /Add topic/ }))
    await waitFor(() => expect(m.createTopic).toHaveBeenCalledWith({ name: 'Rail', query: 'railway delay' }))
  })

  it('keeps Add topic disabled until both fields are filled, and shows the server error when it fails', async () => {
    m.createTopic.mockRejectedValue(new Error('A topic with this name already exists.'))
    render(<Topics />)
    const add = await screen.findByRole('button', { name: /Add topic/ })
    expect((add as HTMLButtonElement).disabled).toBe(true)
    fireEvent.change(screen.getByPlaceholderText('Name (e.g. Fuel prices)'), { target: { value: 'Fuel prices' } })
    fireEvent.change(screen.getByPlaceholderText('Search query'), { target: { value: 'petrol' } })
    fireEvent.click(add)
    expect(await screen.findByText('A topic with this name already exists.')).toBeTruthy()
  })

  it('shows alerts and drift for the selected topic and acknowledges an alert', async () => {
    m.ackAlert.mockResolvedValue({ ...ALERT, acknowledged: true })
    await openTopic()
    expect(await screen.findByText('Negative share rose from 20% to 40%.')).toBeTruthy()
    expect(await screen.findByText('35.0% vs 33.4%')).toBeTruthy() // unknown-word rate, window vs baseline
    expect(screen.getByText('0.31')).toBeTruthy() // PSI
    fireEvent.click(screen.getByRole('button', { name: 'Acknowledge' }))
    await waitFor(() => expect(m.ackAlert).toHaveBeenCalledWith(7))
  })

  it('renders collected post text as text, never as HTML, and labels abstentions', async () => {
    m.posts.mockResolvedValue({
      total: 2, page: 1, limit: 15, total_pages: 1, posts: [
        { id: 1, source: 'google_news', url: 'https://news.example/1', text: '<img src=x onerror=alert(1)> price hike',
          prediction: 'negative', confidence: 0.9, decision_source: 'local:linearsvc', published_at: null },
        { id: 2, source: 'google_news', url: null, text: 'unsure about this one', prediction: 'uncertain',
          confidence: 0.4, decision_source: 'local:linearsvc', published_at: null },
      ],
    })
    const { container } = render(<Topics />)
    fireEvent.click(await screen.findByRole('button', { name: /Fuel prices/ }))
    fireEvent.click(await screen.findByRole('button', { name: 'posts' }))
    expect(await screen.findByText('<img src=x onerror=alert(1)> price hike')).toBeTruthy()
    expect(container.querySelector('img')).toBeNull()
    expect(within(screen.getByText('unsure about this one').parentElement!).getByText('Uncertain')).toBeTruthy()
    expect(m.posts).toHaveBeenCalledWith(1, { sentiment: undefined, page: 1, limit: 15 })
  })

  it('filters posts by label', async () => {
    render(<Topics />)
    fireEvent.click(await screen.findByRole('button', { name: /Fuel prices/ }))
    fireEvent.click(await screen.findByRole('button', { name: 'posts' }))
    await waitFor(() => expect(m.posts).toHaveBeenCalled())
    fireEvent.change(screen.getByDisplayValue('All labels'), { target: { value: 'uncertain' } })
    await waitFor(() => expect(m.posts).toHaveBeenLastCalledWith(1, { sentiment: 'uncertain', page: 1, limit: 15 }))
  })

  it('shows the server message when collecting is refused', async () => {
    m.ingestTopic.mockRejectedValue(new Error('Invalid or missing key. Send it as X-API-Key.'))
    await openTopic()
    fireEvent.click(screen.getByRole('button', { name: /Collect now/ }))
    expect(await screen.findByText('Invalid or missing key. Send it as X-API-Key.')).toBeTruthy()
  })

  it('shows themes and the daily note, or says no LLM is configured', async () => {
    m.insights.mockResolvedValue({
      llm: { configured: false, provider: null, model: null }, digest: null,
      themes: { window_hours: 72, n_posts: 40, unclustered: 0, budget_exhausted: false, created_at: '2026-10-30T08:00:00', themes: [
        { label: 'crude / petrol', summary: '', quotes: [], sentiment: 'negative', share: 0.6, n: 24, llm: false, post_ids: [3] }] },
    })
    await openTopic()
    fireEvent.click(screen.getByRole('button', { name: 'themes' }))
    expect(await screen.findByText('crude / petrol')).toBeTruthy()
    expect(screen.getByText(/No LLM configured/)).toBeTruthy()
  })
})

describe('Agent', () => {
  it('starts a session, shows progress, then the verified answer with its cited posts', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    m.startAgent.mockResolvedValue({ id: 5, status: 'running' })
    const step = { step: 1, action: 'tool' as const, tool: 'search_posts', args: { query: 'petrol' }, result: '{"posts":[]}' }
    const base = { id: 5, kind: 'ask' as const, question: 'q', topic_id: null, error: null, tokens_in: 100, tokens_out: 40,
      latency_ms: 1200, provider: 'groq', model: 'llama', }
    m.agentRun
      .mockResolvedValueOnce({ ...base, status: 'running', steps: [step], answer: null })
      .mockResolvedValue({ ...base, status: 'done', steps: [step, { step: 2, action: 'final' }], answer: {
        text: 'People are unhappy about petrol prices [9].', citations: [9], removed_citations: [4242], verified: true,
        sources: [{ id: 9, text: 'petrol price hike hurts commuters', sentiment: 'negative', url: null, source: 'google_news' }] } })

    render(<Agent />)
    await screen.findByRole('option', { name: 'Fuel prices' })
    const run = screen.getByRole('button', { name: /Run/ })
    expect((run as HTMLButtonElement).disabled).toBe(true) // an ask needs a question
    fireEvent.change(screen.getByPlaceholderText(/Why did negativity/), { target: { value: 'What about petrol?' } })
    fireEvent.click(run)

    expect(await screen.findByText(/search_posts/)).toBeTruthy()
    expect(m.startAgent).toHaveBeenCalledWith({ kind: 'ask', question: 'What about petrol?', topic_id: null })
    await vi.advanceTimersByTimeAsync(1600)
    expect(await screen.findByText('People are unhappy about petrol prices [9].')).toBeTruthy()
    expect(screen.getByText('petrol price hike hurts commuters')).toBeTruthy()
    expect(screen.getByText(/Removed citations with no matching retrieved post: 4242/)).toBeTruthy()
    expect(screen.queryByText(/could not be checked/)).toBeNull()
  })

  it('warns when an answer cites nothing, and shows start errors from the server', async () => {
    m.startAgent.mockRejectedValueOnce(new Error('Protected endpoints are disabled: set INGEST_TOKEN or create an API key.'))
    render(<Agent />)
    fireEvent.change(await screen.findByPlaceholderText(/Why did negativity/), { target: { value: 'q' } })
    fireEvent.click(screen.getByRole('button', { name: /Run/ }))
    expect(await screen.findByText(/Protected endpoints are disabled/)).toBeTruthy()

    m.startAgent.mockResolvedValue({ id: 6, status: 'running' })
    m.agentRun.mockResolvedValue({ id: 6, kind: 'ask', question: 'q', topic_id: null, status: 'done', steps: [{ step: 1, action: 'final' }],
      error: null, tokens_in: 1, tokens_out: 1, latency_ms: 10, provider: 'groq', model: 'x',
      answer: { text: 'Trust me.', citations: [], removed_citations: [], sources: [], verified: false } })
    fireEvent.click(screen.getByRole('button', { name: /Run/ }))
    expect(await screen.findByText(/could not be checked against the data/)).toBeTruthy()
  })

  it('a report needs a topic', async () => {
    render(<Agent />)
    await screen.findByRole('option', { name: 'Fuel prices' })
    fireEvent.change(screen.getAllByRole('combobox')[0], { target: { value: 'report' } })
    expect((screen.getByRole('button', { name: /Run/ }) as HTMLButtonElement).disabled).toBe(true)
    fireEvent.change(screen.getAllByRole('combobox')[1], { target: { value: '1' } })
    expect((screen.getByRole('button', { name: /Run/ }) as HTMLButtonElement).disabled).toBe(false)
  })
})

describe('Annotate', () => {
  const ITEM = { id: 1, post_id: 11, topic_id: 1, text: 'hard to call this one', url: null, reason: 'uncertain' as const, priority: 1 }

  it('lists the queue, adds unsure posts for a topic and exports Label Studio tasks', async () => {
    m.annotationQueue.mockResolvedValue([ITEM])
    m.refreshQueue.mockResolvedValue({ queued: 3 })
    m.exportQueue.mockResolvedValue([{ data: { text: 'hard to call this one', id: 'post-11' } }])
    URL.createObjectURL = vi.fn(() => 'blob:x')
    URL.revokeObjectURL = vi.fn()
    render(<Annotate />)
    expect(await screen.findByText('hard to call this one')).toBeTruthy()
    expect(screen.getByText('model abstained')).toBeTruthy()

    const add = screen.getByRole('button', { name: /Add unsure posts/ }) as HTMLButtonElement
    expect(add.disabled).toBe(true) // needs a topic
    await screen.findByRole('option', { name: 'Fuel prices' })
    fireEvent.change(screen.getAllByRole('combobox')[0], { target: { value: '1' } })
    await waitFor(() => expect(add.disabled).toBe(false))
    fireEvent.click(add)
    expect(await screen.findByText('Queued 3 more posts.')).toBeTruthy()
    expect(m.refreshQueue).toHaveBeenCalledWith(1)

    fireEvent.click(screen.getByRole('button', { name: /Label Studio JSON/ }))
    expect(await screen.findByText('Exported 1 tasks.')).toBeTruthy()
    expect(URL.createObjectURL).toHaveBeenCalled()
  })

  it('shows an empty queue', async () => {
    m.annotationQueue.mockResolvedValue([])
    render(<Annotate />)
    expect(await screen.findByText('The queue is empty')).toBeTruthy()
    expect((screen.getByRole('button', { name: /Label Studio JSON/ }) as HTMLButtonElement).disabled).toBe(true)
  })
})

describe('PostBadge', () => {
  it('distinguishes an abstention from a sentiment', () => {
    render(<><PostBadge label="uncertain" /><PostBadge label="positive" /></>)
    const row = screen.getByText('Uncertain').parentElement!
    expect(within(row.parentElement!).getByText('Positive')).toBeTruthy()
  })
})
