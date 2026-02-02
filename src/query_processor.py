"""Query processing and context management for RAG assistant."""

import config as config_mod
import file_utils as file_utils_mod
import logger as logger_mod

# Re-export constants for backwards compatibility within this module
DISTANCE_THRESHOLD = config_mod.DISTANCE_THRESHOLD
MEMORY_KEY_PARAM = config_mod.CHAT_HISTORY
QUERY_AUGMENTATION_PATH = config_mod.QUERY_AUGMENTATION_PATH

logger = logger_mod.logger


class QueryProcessor:
    """Handles query augmentation, context building, and validation."""

    def __init__(self, memory_manager=None, llm=None):
        """Initialize the query processor."""
        self.memory_manager = memory_manager
        self.llm = llm
        self.follow_up_keywords = []
        self.new_topic_keywords = []
        self._load_augmentation_config()

    def _load_augmentation_config(self) -> None:
        """Load query augmentation configuration from YAML file."""
        try:
            config = file_utils_mod.load_yaml(QUERY_AUGMENTATION_PATH)
            self.follow_up_keywords = config.get("follow_up_keywords", [])
            self.new_topic_keywords = config.get("new_topic_keywords", [])
            logger.info("Query augmentation configuration loaded successfully.")
        except FileNotFoundError:
            logger.warning("query-augmentation.yaml not found. Using default keywords.")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Error loading augmentation config: {e}. Using defaults.")

    def augment_query_with_context(self, query: str) -> str:
        """
        Augment the user's query with recent conversation context.
        This helps resolve pronouns and references in follow-up questions.
        However, for new topic questions or very different queries, limit augmentation.

        Args:
            query: Original user query

        Returns:
            Augmented query that includes recent context (if appropriate)
        """
        if not self.memory_manager:
            return query

        try:
            memory_vars = self.memory_manager.get_memory_variables()
            chat_history = memory_vars.get(MEMORY_KEY_PARAM, "")

            if chat_history and chat_history != "No previous conversation context.":
                # Check if this looks like a follow-up question or new topic
                # Keywords are loaded from config/query-augmentation.yaml
                query_lower = query.lower()
                is_follow_up = any(keyword in query_lower for keyword in self.follow_up_keywords)
                is_new_topic = any(keyword in query_lower for keyword in self.new_topic_keywords)

                # If question starts with "what are/is" or similar, it's likely a new topic
                # unless it contains follow-up keywords
                if is_new_topic and not is_follow_up:
                    # New topic question - return without augmentation
                    logger.debug("New topic question detected - using original query for clean search")
                    return query

                if is_follow_up:
                    # For follow-ups, include only the last user question to avoid biasing search
                    # Extract the last user message from chat history
                    lines = chat_history.strip().split("\n")
                    last_user_question = None
                    for line in reversed(lines):
                        if line.startswith("Human: "):
                            last_user_question = line[7:].strip()  # Remove 'Human: ' prefix
                            break

                    if last_user_question:
                        augmented_query = f"Previous question: {last_user_question}\n\nCurrent question: {query}"
                        logger.debug("Query augmented with last user question for follow-up")
                        return augmented_query

                    # Fallback to full context if we can't extract
                    augmented_query = f"{chat_history}\n\nCurrent question: {query}"
                    logger.debug("Query augmented with full chat history context (fallback)")
                    return augmented_query

                # Default: treat as new topic for clean search
                logger.debug("Query treated as new topic - using original query for clean search")
                return query

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(f"Could not augment query with context: {e}")

        return query

    @staticmethod
    def build_context(flat_docs: list[str], flat_distances: list[float]) -> str:
        """Assemble a context string only if the top result meets the distance threshold."""
        if flat_docs and (not flat_distances or flat_distances[0] <= DISTANCE_THRESHOLD):
            return "\n".join(flat_docs)
        return ""

    def validate_context(self, query: str, context: str, search_query: str = None) -> bool:
        """Return True if the context is relevant; False otherwise (conservative on error)."""
        try:
            return self._is_context_relevant_to_query(query, context, search_query)
        except Exception as e:
            logger.warning(f"Context relevance validation failed: {e}")
            return False

    def _is_context_relevant_to_query(self, query: str, context: str, search_query: str = None) -> bool:
        """
        Validate that retrieved context actually addresses the user's query.
        Uses the LLM to perform semantic validation.

        Args:
            query: User's question
            context: Retrieved document context
            search_query: Augmented search query (optional, for follow-up validation)

        Returns:
            True if context answers the query, False otherwise
        """
        if not context or not context.strip():
            return False

        if not self.llm:
            logger.warning("LLM not available for context validation, assuming relevant")
            return True

        try:
            # For follow-up questions, use the search query context for better validation
            if search_query and search_query != query:
                # This is a follow-up question with augmented context
                validation_prompt = (
                    f"Given the conversation context, does the following retrieved context "
                    f"contain information that directly addresses the user's follow-up question?\n\n"
                    f"Conversation: {search_query}\n\n"
                    f"Retrieved Context: {context[:500]}\n\n"
                    f"Answer with only 'YES' or 'NO'."
                )
            else:
                # Regular question validation
                validation_prompt = (
                    f"Does the following context contain information that directly "
                    f"addresses this question? Answer with only 'YES' or 'NO'.\n\n"
                    f"Question: {query}\n\n"
                    f"Context: {context[:500]}"  # Limit to first 500 chars for speed
                )

            result = self.llm.invoke(validation_prompt)
            # Handle both string and AIMessage responses
            result_str = str(result) if not isinstance(result, str) else result
            # For AIMessage objects, extract content attribute
            if hasattr(result, "content"):
                result_str = result.content
            is_relevant = "YES" in result_str.upper()
            logger.info(f"Context relevance validation: {is_relevant}")
            return is_relevant
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(f"Error validating context relevance: {e}")
            # On error, assume context is relevant to avoid false negatives
            return True
